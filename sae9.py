## main function for SAE Trajectory Graph Generation.

import torch
import numpy as np
from hooked import TransformerLensTransparentLlm
from sae_lens import SAE, HookedSAETransformer
import os
import json
from datetime import datetime
from sae6 import analyze_trajectory_origins
from contextlib import contextmanager
import inspect
from sae6 import analyze_layer_contributions_to_prediction

import inspect
import torch
from transformer_lens import HookedTransformer

def show_block_forward(layer_idx=0):
    model = TransformerLensTransparentLlm(
        model_name="google/gemma-2-2b",
        device="cuda",
        dtype=torch.float32
    )
    block = model._model.blocks[layer_idx]
    print(f"Block type: {type(block)}")

    # 1) Путь к файлу с исходником класса блока (полезно проверить)
    try:
        src_file = inspect.getsourcefile(block.__class__) or inspect.getfile(block.__class__)
        print(f"\nSource file: {src_file}")
    except Exception as e:
        print(f"\nCannot get source file: {e}")

    # 2) Сам forward блока
    print("\n--- BLOCK forward ---")
    try:
        fwd = block.forward
        # для bound-method нужен __func__
        if hasattr(fwd, "__func__"):
            code = inspect.getsource(fwd.__func__)
        else:
            code = inspect.getsource(fwd)
        print(code)
    except Exception as e:
        print(f"Cannot get block.forward source: {e}")

    # 3) На случай, если (2) не сработал — вывести код всего класса (включая forward)
    if 'code' not in locals():
        print("\n--- CLASS source (fallback) ---")
        try:
            print(inspect.getsource(block.__class__))
        except Exception as e:
            print(f"Cannot get class source: {e}")

    # 4) Аналогично можно вывести подкомпоненты
    for name in ["ln1","attn","ln1_post","mlp","ln2"]:
        if hasattr(block, name):
            sub = getattr(block, name)
            print(f"\n--- {name}.forward ---")
            try:
                fwd = sub.forward
                code = inspect.getsource(fwd.__func__ if hasattr(fwd,"__func__") else fwd)
                print(code)
            except Exception as e:
                print(f"Cannot get {name}.forward source: {e}")

    return model


@contextmanager
def sae_on_cuda(sae):
    """
    Context manager для временного перемещения SAE на CUDA.
    Автоматически возвращает SAE на исходное устройство после использования.
    """
    original_device = str(sae.device)
    try:
        sae.to("cuda")
        yield sae
    finally:
        sae.to(original_device)




def load_sae(layer_idx):


    if layer_idx != 3:
        release = "gemma-scope-2b-pt-res-canonical"                              
        sae_id = f"layer_{layer_idx}/width_65k/canonical" 

        sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
            release=release, sae_id=sae_id, device="cpu"
        )
    else:
        release = "gemma-scope-2b-pt-res"
        sae_id = "layer_3/width_65k/average_l0_193"
        sae = SAE.from_pretrained(release, sae_id, device="cpu")


    return sae

def run_sae(sae, residual, k_top):
    """
    Run SAE and get top k indices of features.
    """

    _, sae_cache = sae.run_with_cache(residual)
    acts_post = sae_cache["hook_sae_acts_post"]

    _, top_indices = torch.topk(acts_post, k=k_top)
    top_indices = top_indices.tolist()

    return top_indices
    


def analyze_prompt_length(model, sentence_tokens: list, layers: int, k_top: int):
   
    """
    Returns Data that is basically a set of plots of SAE indices for each length of prompt from max_length to 1. 
    Tracing data[length][layer_idx] we identify trajectories and edges in graph_generator.py
    """
    max_length = len(sentence_tokens)
    data = [[[] for _ in range(layers)] for _ in range(max_length + 1)]
    
        
    for layer_idx in range(layers):

        print(f"      Processing layer {layer_idx}...")
        residual_in = model.residual_in(layer=layer_idx)
        residual_after_attn = model.residual_after_attn(layer=layer_idx)

        sae = load_sae(layer_idx)

        for query_pos in range(1, max_length):
            with sae_on_cuda(sae) as cuda_sae:
                decomposed_attention_q = model.decomposed_attn_gemma(layer=layer_idx, query_pos = query_pos)
                #print("decomp_shape", decomposed_attention_q.shape)
                # ------- without adds
                last_residual_before_attn = residual_in[0, query_pos, :]
                last_residual_after_attn = residual_after_attn[0, query_pos, :]

                top_indices_without_attn = run_sae(cuda_sae, last_residual_before_attn, k_top)

                # ------- with adds
                
                token_adds_to_the_last = decomposed_attention_q.sum(dim=1)
                
                cumulative_adds = torch.cumsum(token_adds_to_the_last, dim=0)
                
                # Применяем RMS нормализацию к cumulative_adds for Gemma 2-b
                block = model._model.blocks[layer_idx]
                w = block.ln1_post.w  # [d_model]
                eps = block.ln1_post.eps  # RMS norm eps
                
                rms = torch.sqrt(cumulative_adds.pow(2).mean(dim=-1, keepdim=True) + eps)  # [n_tokens, 1]
                normalized_cumulative_adds = (cumulative_adds / rms) * w     
                
                print(torch.norm(last_residual_after_attn - (last_residual_before_attn + normalized_cumulative_adds[-1,:])))



                top_indices = run_sae(cuda_sae, last_residual_before_attn + normalized_cumulative_adds, k_top)
                top_indices.insert(0, top_indices_without_attn)

                length = query_pos + 1
                data[length][layer_idx] = top_indices
                # SAE автоматически вернулся на исходное устройство (CPU)
        print(top_indices[-2][:15])
    

    return data  


def main():
    MODEL_NAME = "gemma-2-2b"
    SENTENCE = "The final correct answer: Max, Mark and Smith are in the empty dark room. Smith left. Mark gave flashlight to"
    k_top = 400
    

    print("Starting multi-length SAE trajectory generation...")
    

    model = TransformerLensTransparentLlm(
        model_name=MODEL_NAME, 
        device="cuda",
        dtype=torch.float32
    )

    #model._model.cfg.use_normalization_before_and_after = False

    model.run([SENTENCE])
    initial_tokens = model._last_run.tokens[0].clone()
    
    # Генерируем следующие 5 токенов
    print("Generating next 5 tokens...")
    current_tokens = initial_tokens.clone()
    generated_tokens = []
    
    for i in range(5):
        # Запускаем модель на текущей последовательности
        model.run([model._cached_model.tokenizer.decode(current_tokens)])
        
        # Получаем логиты для следующего токена
        with torch.no_grad():
            logits = model._cached_model(current_tokens.unsqueeze(0))  # Add batch dimension
            next_token_logits = logits[0, -1, :]  # Get logits for the last position
        
        # Получаем наиболее вероятный следующий токен (greedy decoding)
        next_token = torch.argmax(next_token_logits).unsqueeze(0)
        
        # Добавляем к последовательности
        current_tokens = torch.cat([current_tokens, next_token])
        token_str = model._cached_model.tokenizer.decode(next_token.item())
        generated_tokens.append(token_str)
        print(f"  Generated token {i+1}: {repr(token_str)}")
    
    print(f"Generated tokens: {generated_tokens}")
    
    # Создаем расширенное предложение
    extended_sentence = SENTENCE + "".join(generated_tokens)
    print(f"Extended sentence: {extended_sentence}")
    
    
    model.run([SENTENCE])
    tokens_tensor = model._last_run.tokens[0]
    results = analyze_layer_contributions_to_prediction(model, tokens_tensor, 26)


    full_tokens_str = model.tokens_to_strings(tokens_tensor)
    
    print(f"\nFull sentence tokens ({len(full_tokens_str)}):")
    for i, token in enumerate(full_tokens_str):
        print(f"{i}: {repr(token)}")

    max_length = len(full_tokens_str)
    n_layers = 26
    
    # Собираем данные
    save_dir = "att_exp/evolution2"
    os.makedirs(save_dir, exist_ok=True)
    data = analyze_prompt_length(model, full_tokens_str, n_layers, k_top)

    # Сохраняем raw data
    metadata = {
        "metadata": {
            "original_sentence": SENTENCE,
            "max_prompt_length": max_length,
            "k_top": k_top,
            "n_layers": n_layers,
            "tokens": full_tokens_str
        },
        "data": data,
        "results": results
    }
    
    filename = os.path.join(save_dir, f'sae_data.json')
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== COMPLETED ===")
    print(f"Saved SAE data: {filename}")
    print(f"Data collected for {len(data)} different prompt lengths")


if __name__ == "__main__":
    main()