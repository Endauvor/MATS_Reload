from dataclasses import dataclass
from typing import List, Optional
import os

import torch
import transformer_lens
import transformers
from fancy_einsum import einsum
from jaxtyping import Float, Int
from typeguard import typechecked
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch
from sae_lens import SAE, HookedSAETransformer



@dataclass
class _RunInfo:
    tokens: Int[torch.Tensor, "batch pos"]
    logits: Float[torch.Tensor, "batch pos d_vocab"]
    cache: transformer_lens.ActivationCache


def load_hooked_transformer(
    model_name: str,
    hf_model: Optional[transformers.PreTrainedModel] = None,
    tlens_device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    supported_model_name: Optional[str] = None,
):
    
    tlens_model = transformer_lens.HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        fold_ln=False,  # Keep layer norm where it is.
        center_writing_weights=False,
        center_unembed=False,
        device=tlens_device,
        dtype=dtype,
    )
    tlens_model.eval()
    return tlens_model



def load_hooked_transformer_Llama(
    model_name: str,
    hf_model: Optional[transformers.PreTrainedModel] = None,
    tlens_device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    supported_model_name: Optional[str] = None
):
    """
    Loads Llama-3.2-3B-Instruct model and converts it to HookedTransformer
    """

    token = os.getenv("HF_TOKEN", None)  # Set your Hugging Face token as environment variable
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=token,
        torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=token
    )
    
    hf_config = hf_model.config
    
    max_ctx = min(hf_config.max_position_embeddings, 2048)
    
    cfg = HookedTransformerConfig(
        n_layers=hf_config.num_hidden_layers,
        d_model=hf_config.hidden_size,
        d_head=hf_config.hidden_size // hf_config.num_attention_heads,
        n_heads=hf_config.num_attention_heads,
        d_mlp=hf_config.intermediate_size,
        d_vocab=hf_config.vocab_size,
        n_ctx=max_ctx,  
        act_fn=hf_config.hidden_act,  
        model_name=model_name,
        normalization_type="RMS",  
        device=tlens_device
    )
    
    tlens_model = HookedTransformer(cfg)
    
    tlens_model.load_state_dict(hf_model.state_dict(), strict=False)

    tlens_model.tokenizer = tokenizer
    
    tlens_model.eval()
    
    return tlens_model




class TransformerLensTransparentLlm():
   
    def __init__(
        self,
        model_name: str,
        hf_model: Optional[transformers.PreTrainedModel] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        supported_model_name: str = None
    ):
        
        self._model_name = model_name
        self._supported_model_name = supported_model_name

        self.dtype = dtype
        self.hf_tokenizer = tokenizer
        self.hf_model = hf_model
        self._last_run = None
        self.device = "cuda"
        self._cached_model = None
       

   

    @torch.no_grad()
    def run(self, sentences: List[str]) -> None:
        # Use tokenizer directly instead of model.to_tokens
        tokenizer = self._model.tokenizer
        tokens = tokenizer(sentences, return_tensors="pt")["input_ids"]
        tokens = tokens.to(self._model.cfg.device)
        
        logits, cache = self._model.run_with_cache(tokens)

        self._last_run = _RunInfo(
            tokens=tokens,
            logits=logits,
            cache=cache,
        )

    @typechecked
    def residual_in(self, layer: int):# -> Float[torch.Tensor, "batch pos d_model"]:
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_pre")

    @typechecked
    def residual_after_attn(
        self, layer: int
    ): #-> Float[torch.Tensor, "batch pos d_model"]:
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_mid")

    @typechecked
    def residual_after_LN(
        self, layer: int
    ): 
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "ln1.hook_normalized")
        
    @typechecked
    def attention_output(
        self,
        layer: int,
        pos: int,
    ): #> Float[torch.Tensor, "d_model"]:
        return self._get_block(layer, "hook_attn_out")[0][pos]

    def _get_block(self, layer: int, block_name: str) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.cache[f"blocks.{layer}.{block_name}"]
        
    # @typechecked
    def tokens_to_strings(self, tokens: Int[torch.Tensor, "pos"]) -> List[str]:
        # Use tokenizer directly instead of model.to_str_tokens
        tokenizer = self._model.tokenizer
        return tokenizer.convert_ids_to_tokens(tokens.tolist())

    @property
    def _model(self):
        if self._cached_model is None:
            tlens_model = load_hooked_transformer(
                self._model_name,
                hf_model=self.hf_model,
                tlens_device=self.device,
                dtype=self.dtype,
                supported_model_name=self._supported_model_name,
            )

            # If we have a custom tokenizer, override the one from load_hooked_transformer
            if self.hf_tokenizer is not None:
                tlens_model.tokenizer = self.hf_tokenizer

            tlens_model.set_use_attn_result(True)
            
            # Handle Grouped Query Attention models (like Gemma)
            if hasattr(tlens_model.cfg, 'n_key_value_heads') and tlens_model.cfg.n_key_value_heads is not None:
                # For GQA models, use split_qkv_input instead of attn_in
                tlens_model.set_use_split_qkv_input(True)
            else:
                # For regular models, use attn_in
                tlens_model.set_use_attn_in(False)
                tlens_model.set_use_split_qkv_input(False)

            self._cached_model = tlens_model

        return self._cached_model


    @torch.no_grad()
    @typechecked
    def decomposed_attn(
        self, layer: int
    ): # --> ["pos key_pos head d_model"]
        if not self._last_run:
            raise self._run_exception
        
        
        hook_v = self._get_block(layer, "attn.hook_v")[0]
        b_v = self._model.blocks[layer].attn.b_V


        v = hook_v + b_v
        
        pattern = self._get_block(layer, "attn.hook_pattern")[0].to(v.dtype)

        z = einsum(
            "key_pos head d_head, "
            "head query_pos key_pos -> "
            "query_pos key_pos head d_head",
            v,
            pattern,
        )

        decomposed_attn = einsum(
            "pos key_pos head d_head, "
            "head d_head d_model -> "
            "pos key_pos head d_model",
            z,
            self._model.blocks[layer].attn.W_O,
        )

        

        _, n_key, n_head, d_model = decomposed_attn.shape
        b_O = self._model.blocks[layer].attn.b_O
        b_O_adds = (b_O / (n_key * n_head)).view(1, 1, 1, d_model)



        return decomposed_attn + b_O_adds


    @torch.no_grad()
    @typechecked
    def decomposed_attn_q(
        self, layer: int, 
        query_pos: int
    ): # --> ["pos key_pos head d_model"]
        if not self._last_run:
            raise self._run_exception
        
        hook_v = self._get_block(layer, "attn.hook_v")[0]
        b_v = self._model.blocks[layer].attn.b_V

        # support for gqa
        num_head_groups = b_v.shape[-2] // hook_v.shape[-2]
        
        hook_v = hook_v.repeat_interleave(num_head_groups, dim=-2)
        
        v = hook_v + b_v
        
        pattern = self._get_block(layer, "attn.hook_pattern")[0].to(v.dtype)

        z = einsum(
            "key_pos head d_head, "
            "head query_pos key_pos -> "
            "query_pos key_pos head d_head",
            v,
            pattern,
        )

        decomposed_attn = einsum(
            "pos key_pos head d_head, "
            "head d_head d_model -> "
            "pos key_pos head d_model",
            z,
            self._model.blocks[layer].attn.W_O,
        )

        decomposed_attn_q = decomposed_attn[query_pos, ...]

        _, n_head, d_model = decomposed_attn_q.shape
        b_O = self._model.blocks[layer].attn.b_O
        b_O_adds = (b_O / ((query_pos + 1) * n_head)).view(1, 1, d_model)



        return decomposed_attn_q[:query_pos+1,...] + b_O_adds

    
        

    @torch.no_grad()
    @typechecked
    def decomposed_attn_gemma(
        self, layer: int, 
        query_pos: int
    ): # --> ["pos key_pos head d_model"]
        if not self._last_run:
            raise self._run_exception
        
        hook_v = self._get_block(layer, "attn.hook_v")[0] # [25, 4, 256]
        b_v = self._model.blocks[layer].attn.b_V #b_v shape: [8, 256] 
        

        # Support for Grouped Query Attention (GQA) - expand hook_v to match all heads
        num_head_groups = b_v.shape[-2] // hook_v.shape[-2]
        hook_v = hook_v.repeat_interleave(num_head_groups, dim=-2)

        
        v = hook_v
        
        pattern = self._get_block(layer, "attn.hook_pattern")[0].to(v.dtype)

        z = einsum(
            "key_pos head d_head, "
            "head query_pos key_pos -> "
            "query_pos key_pos head d_head",
            v,
            pattern,
        )

        decomposed_attn = einsum(
            "pos key_pos head d_head, "
            "head d_head d_model -> "
            "pos key_pos head d_model",
            z,
            self._model.blocks[layer].attn.W_O,
        )

        decomposed_attn_q = decomposed_attn[query_pos, ...]

        return decomposed_attn_q[:query_pos+1,...]

    def _apply_rms_norm(self, x, norm_layer, eps=1e-6):
        """Применяет RMS нормализацию как в Gemma"""
        # x shape: [seq_len, n_heads, d_model]
        
        # Стандартная RMS нормализация
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        
        # Применяем веса RMSNorm (в TransformerLens это параметр 'w')
        if hasattr(norm_layer, 'w') and norm_layer.w is not None:
            x = x * norm_layer.w
            
        return x

        