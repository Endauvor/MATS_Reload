"""
QK curcuit analysis, adapting Anthropic approach.
Finds SAE to SAE interaction for a given attention score. 
All biases and error terms are divided equally and added to each SAE feature.
The same is true for LN layer. 
As a result after LN layer we have a set of 'SAE' features sum of which with activations is equal to hidden representation. 
I checked all in great detail and there is not mistakes in accuracy. 
"""

import torch
from hooked import TransformerLensTransparentLlm
from contribute import get_attention_contributions
from sae_lens import SAE, HookedSAETransformer
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import requests


def save_activation(store: dict, store_key: str):
    """Factory for a hook that saves an activation to a dictionary."""
    def _fn(tensor, hook):
        store[store_key] = tensor.detach().clone()
    return _fn

def save_scores(store: dict, store_key: str):
    def _fn(tensor, hook):
        store[store_key] = tensor.detach().clone()  # [batch, head, q, k]
    return _fn

def save_pattern(store: dict, store_key: str):
    def _fn(tensor, hook):
        store[store_key] = tensor.detach().clone()  # [batch, head, q, k]
    return _fn

def analyze_sae_features(features: torch.Tensor, top_k: int, description: str, compare_indices=None):
    """Analyzes and prints top-K most active SAE features from ready activation tensor."""
    top_values, top_indices = torch.topk(features, k=top_k)
    
    print(f"\n--- Top {top_k} SAE Features for: {description} ---")
    print("Feature Index | Activation Value")
    print("----------------|-----------------")
    for val, idx in zip(top_values, top_indices):
        if val.item() > 1e-4: # Print only those that are actually active
            print(f"{idx.item():<15} | {val.item():.4f}")
    
    # If comparison indices provided, show new features and common features
    if compare_indices is not None:
        current_indices = set(top_indices.cpu().numpy())
        compare_indices_set = set(compare_indices.cpu().numpy())
        
        # New features (in current but not in comparison)
        new_features = current_indices - compare_indices_set
        if new_features:
            print(f"\n--- New features in {description} (not in comparison) ---")
            print("Position | Feature Index | Activation Value")
            print("---------|---------------|------------------")
            
            # Get positions and values for new features
            new_features_with_pos = []
            for pos, idx in enumerate(top_indices.cpu().numpy()):
                if idx in new_features:
                    val = top_values[pos]
                    new_features_with_pos.append((pos + 1, idx, val.item()))  # pos+1 for 1-based indexing
            
            # Sort by position
            for pos, idx, val in sorted(new_features_with_pos):
                print(f"{pos:<8} | {idx:<13} | {val:.4f}")
        else:
            print(f"\n--- No new features in {description} ---")
        
        # Common features (intersection)
        common_features = current_indices & compare_indices_set
        if common_features:
            print(f"\n--- Common features in both lists ---")
            print("Position | Feature Index | Activation Value")
            print("---------|---------------|------------------")
            
            # Get positions and values for common features
            common_features_with_pos = []
            for pos, idx in enumerate(top_indices.cpu().numpy()):
                if idx in common_features:
                    val = top_values[pos]
                    common_features_with_pos.append((pos + 1, idx, val.item()))
            
            # Sort by position
            for pos, idx, val in sorted(common_features_with_pos):
                print(f"{pos:<8} | {idx:<13} | {val:.4f}")
        else:
            print(f"\n--- No common features found ---")
    
    return top_indices

def plot_attention_heatmap(scores: torch.Tensor, tokens_str: list, title: str, filename: str, is_diff: bool = False, show_values: bool = True):
    """
    Draws and saves attention scores heatmap.

    Args:
        scores (torch.Tensor): Attention scores tensor [head, key_pos]
        tokens_str (list): List of tokens for X axis.
        title (str): Plot title.
        filename (str): Path to save the file.
        is_diff (bool): If True, centers color scale on 0 for difference display.
        show_values (bool): If True, displays numerical values in each cell.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    scores_np = scores.detach().cpu().numpy()
    n_heads, n_tokens = scores_np.shape
    
    fig, ax = plt.subplots(figsize=(max(6, n_tokens / 1.5), max(4, n_heads / 2)))
    
    # Configure color scale
    if is_diff:
        v_max = np.abs(scores_np).max()
        vmin, vmax = -v_max, v_max
        cmap = 'coolwarm'  # Blue < 0, White = 0, Red > 0
    else:
        vmin, vmax = 0, scores_np.max()
        cmap = 'coolwarm'  # From blue (0) to red (max)
        
    # Create heatmap
    cax = ax.imshow(scores_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add numerical values to cells
    if show_values:
        for i in range(n_heads):
            for j in range(n_tokens):
                value = scores_np[i, j]
                # Choose text color based on background brightness
                text_color = 'white' if abs(value - (vmax + vmin) / 2) > (vmax - vmin) * 0.3 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                       color=text_color, fontsize=8, weight='bold')
    
    # Configure axes
    ax.set_xticks(np.arange(n_tokens))
    ax.set_xticklabels(tokens_str, rotation=90)
    ax.set_yticks(np.arange(n_heads))
    ax.set_xlabel("Token Position (Key)")
    ax.set_ylabel("Attention Head")
    
    # Set title
    ax.set_title(title)
    
    # Add color bar
    fig.colorbar(cax, label="Attention Score")
    
    # Save and close
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    #print(f"Heatmap saved")

def plot_projected_contributions_heatmap_per_head(
    sae_decomposition_last: torch.Tensor, 
    sae_decomposition_k: torch.Tensor, 
    W_Q: torch.Tensor, 
    W_K: torch.Tensor,
    b_Q: torch.Tensor,
    b_K: torch.Tensor,
    head_idx: int,
    k_top: int, 
    token_idx1: int, 
    token_idx2: int, 
    filename: str,
    scale: float,
    top_indices: torch.Tensor,
    layer_idx: int,
    show_values: bool = True
):
    """
    Draws heatmap of pairwise dot products of feature contribution projections for ONE HEAD.
    contributions: [top_pos, d_model]
    W_Q, W_K: [d_model, d_head] (already for specific head)
    b_Q, b_K: [d_head] (already for specific head)
    show_values: If True, displays numerical values in each cell.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Project contributions to Q and K spaces and add bias
    K = sae_decomposition_last.shape[0]
    q_proj = torch.einsum("tm,md->td", sae_decomposition_last, W_Q) + b_Q / K
    k_proj = torch.einsum("tm,md->td", sae_decomposition_k, W_K) + b_K / K
    
    # Calculate pairwise dot products
    dot_products = torch.einsum("id,jd->ij", q_proj, k_proj) / (W_Q.size(-1) ** 0.5)
    total_logit_contribution = dot_products.sum().item()
    print("approximate score:", total_logit_contribution)
    dot_products_np = dot_products.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    v_max = np.abs(dot_products_np).max()
    cax = ax.imshow(dot_products_np, cmap='coolwarm', vmin=-v_max, vmax=v_max)
    
    # Add numerical values to cells
    if show_values:
        for i in range(k_top):
            for j in range(k_top):
                value = dot_products_np[i, j]
                # Choose text color based on background brightness
                text_color = 'white' if abs(value) > v_max * 0.5 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                       color=text_color, fontsize=7, weight='bold')
    
    # sae_name = f"{layer_idx-1}-res_post_32k-oai"

    # def get_feature_label(feature_idx):
    #     try:
    #         url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{sae_name}/{feature_idx}"
    #         response = requests.get(
    #             url,
    #             headers={"x-api-key": "sk-np-mr5bqLiLmlKbpioVXmplW5m0BkNmtBSEdTsruyS55HA0"},
    #             timeout=5
    #         )
    #         if response.status_code == 200:
    #             data = response.json()
    #             pos_str = data.get("pos_str", [])
    #             return pos_str[0] if pos_str else "N/A"
    #         else:
    #             return "API Err"
    #     except Exception:
    #         return "Req Err"

    token1_feature_indices = top_indices[token_idx1].cpu().tolist()
    y_labels = [str(idx) for idx in token1_feature_indices]
            
    token2_feature_indices = top_indices[token_idx2].cpu().tolist()
    x_labels = [str(idx) for idx in token2_feature_indices]

    ax.set_xticks(np.arange(k_top))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(k_top))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel(f"Top Features of Token {token_idx2} (Key)")
    ax.set_ylabel(f"Top Features of Token {token_idx1} (Query)")
    ax.set_title(f"SAE Interaction: Top {k_top} Features for Tokens {token_idx2}(K) & {token_idx1}(Q)\nTotal Logit Contribution: {total_logit_contribution:.4f}")
    
    fig.colorbar(cax, label="Projected Dot Product")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    #print(f"Projected contributions heatmap for head {head_idx} saved.")




def ln_decompose_parts(
    model,
    X_parts: torch.Tensor,      # [top k, token_pos, d_model] — contributions BEFORE LN
    layer_idx: int, 
):
    """
    Exact additive decomposition of LayerNorm:
      LN(x)_t = sum_k [ gamma * (X_k - mean(X_k)) / sigma(x) ] + beta / k_top (distribute beta across parts)

    Returns:
      LN_parts: [K, T, D] — contribution of each part after LN
      LN_sum:   [T, D]    — sum of contributions (should match LN_true)
      LN_true:  [T, D]    — direct LN(x_total) for verification
    """

    w = model._model.blocks[layer_idx].ln1.w           # [d]  (γ)
    b = model._model.blocks[layer_idx].ln1.b           # [d]  (β)
    eps  = getattr(model._model.blocks[layer_idx].ln1, "eps", 1e-5)

    K, _, _ = X_parts.shape
    
    device, dtype = X_parts.device, X_parts.dtype
    w = w.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)

    # x_total and LN statistics for each token (t)
    x_total = X_parts.sum(dim=0)                       # [T, D]
    mu  = x_total.mean(dim=-1, keepdim=True)           # [T, 1]
    xc  = x_total - mu                                 # [T, D]
    var = (xc * xc).mean(dim=-1, keepdim=True)         # [T, 1]
    sigma = torch.sqrt(var + eps)                      # [T, 1]

    # mean of EACH part over D
    mu_parts = X_parts.mean(dim=-1, keepdim=True)      # [K, T, 1]

    # contributions after LN (without beta)
    LN_parts = (X_parts - mu_parts) / sigma.unsqueeze(0)   # [K, T, D]
    LN_parts = LN_parts * w.view(1, 1, -1)                 # gamma

   
    LN_parts = LN_parts + b.view(1, 1, -1) / K
    
    LN_sum  = LN_parts.sum(dim=0)                                      # [T, D]
    LN_true = ((x_total - mu) / sigma) * w + b                         # [T, D]

    return LN_parts, LN_sum, LN_true



def get_sae_decomposition(sae, sae_cache_pre, k_top, scale_k, k, residual_pre, errors_out):

    acts_post = sae_cache_pre["hook_sae_acts_post"] # [n_tokens, d_sae]
    top_vals, top_idx = torch.topk(acts_post, k=k_top, dim=-1)
    topk_sae_features = sae.W_dec[top_idx]
    print(top_idx[5,:])

    


    topk_sae_features = topk_sae_features.permute(1, 0, 2).contiguous()  # [top_pos, token_pos, d_model]
    topk_sae_activations = top_vals.permute(1, 0).contiguous()           # [top_pos, token_pos]

    reconstruction_with_topk = topk_sae_activations.unsqueeze(-1) * topk_sae_features
    reconstruction_with_topk = reconstruction_with_topk.sum(dim = 0)
    reconstruction_with_topk = (reconstruction_with_topk + sae.b_dec)* sae.ln_std +  sae.ln_mu
    errors = residual_pre - reconstruction_with_topk 

    topk_sae_features[0, k, :] = topk_sae_features[0, k, :] * scale_k

    sae_decompose = (topk_sae_activations.unsqueeze(-1) * topk_sae_features \
        + sae.b_dec / k_top) * sae.ln_std + sae.ln_mu / k_top + errors / k_top  


    return sae_decompose, top_idx



# has not been finished
@torch.no_grad()
def decompose_values_per_head_simple(
    ln_parts: torch.Tensor,         # [K, T, D] — sum over K ≈ ln1.hook_normalized
    W_V: torch.Tensor,              # [H, D, d_head]
    b_V: torch.Tensor,              # [H, d_head]
    pattern_last: torch.Tensor,     # [H, T] = pattern_after[:, -1, :]
):
    """
    Returns:
      v_parts: [K, H, T, d_head] — contribution of each part k to value of each head h and token t
      v_sum:   [T, H, d_head]    — sum over parts (should match hook_v)
    """
    K, T, D = ln_parts.shape
    H, D2, E = W_V.shape
    assert D == D2, f"D mismatch: {D} vs {D2}"

    v_parts = torch.einsum('ktd,hde->khte', ln_parts, W_V)         # [K,H,T,E]

    # uniformly distribute bias V across K parts
    v_parts = v_parts + b_V.to(v_parts.dtype).to(v_parts.device).view(1, H, 1, E) / K

    attn_w = pattern_last.to(v_parts.dtype).to(v_parts.device).view(1, H, T, 1)  # [1,H,T,1]

    v_parts_weighted = v_parts * attn_w                    # [K,H,T,E]
    # sum over parts → full V
    v_sum = v_parts.sum(dim=0).permute(1, 0, 2).contiguous()       # [T,H,E]

    return v_parts, v_sum



logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("SAE")

def log_vector_diff(name: str, v1: torch.Tensor, v2: torch.Tensor, n_show: int = 10):
    diff_norm = torch.norm(v1 - v2).item()
    logger.info(
        f"{name}: ||v1 - v2|| = {diff_norm:.4f}\n"
        f"  v1[:{n_show}] = {v1[:n_show].detach().cpu().numpy()}\n"
        f"  v2[:{n_show}] = {v2[:n_show].detach().cpu().numpy()}"
    )



def main():

    MODEL_NAME = "gpt2-small"
    #SENTENCE = "The final correct answer: Mark, Max and Smith are in the empty dark room. Smith left. Mark gave flashlight to"
    SENTENCE = "The blues is a genre of music that originated in the Deep South in the 1920s. It is characterized by its soulful and emotive sound, often featuring lyrics that express feelings of sadness, loss, and longing. The blues genre has evolved over the years, incorporating various styles and influences, but its core elements remain the same. From Robert Johnson to B.B. King, legendary blues musicians have made significant contributions to the genre. The blues has also had a profound impact on other genres, such as rock and roll, jazz, and rhythm and blues. Today, the blues remains a popular and enduring genre of music, with many artists continuing to create and perform blues music. The blues is a universal language, expressing emotions and experiences that transcend borders and cultures. The final correct answer: Max, Mark and Smith are in the empty dark room. Smith left. Mark gave flashlight to"
    
    layer_idx = 5     
    k = 5            
    scale_k = 100
    device = "cuda"
    dtype = torch.float32   
    k_top = 10

    model = TransformerLensTransparentLlm(
        model_name=MODEL_NAME,
        device=device,
        dtype=dtype,
    )
    model.run([SENTENCE])

    tokens = model._last_run.tokens  
    tokens_tensor = model._last_run.tokens[0] 
    tokens_str = model.tokens_to_strings(tokens_tensor)
    
    print("\n Tokens: ")
    for i, token in enumerate(tokens_str):
        print(f"{i}: {repr(token)}")

    #d_model = model._model.cfg.d_model

    release = "gpt2-small-resid-post-v5-32k"
    sae_id  = f"blocks.{layer_idx-1}.hook_resid_post"
    sae, cfg, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release, sae_id=sae_id, device=device
    )
    sae.use_error_term = False 

    resid_pre_all = model.residual_in(layer=layer_idx)[0].to(device)
    _, sae_cache_pre = sae.run_with_cache(resid_pre_all)
    sae_out  = sae_cache_pre["hook_sae_output"]
    errors = resid_pre_all - sae_out

    print(cfg)
    
    sae_decompose, top_indices = get_sae_decomposition(sae, sae_cache_pre, k_top, scale_k, k, resid_pre_all, errors)
    print(top_indices[k,:])

    sae_decompose_k = sae_decompose[:, k, :]
    get_back_k = sae_decompose_k.sum(dim = 0)

    print(resid_pre_all.shape)
    log_vector_diff("sae_decomposed_k resid_pre_k", get_back_k, resid_pre_all[k,:])
    
    sae_decompose_LN, _, _ = ln_decompose_parts(model, sae_decompose, layer_idx)

    get_back = sae_decompose_LN.sum(dim = 0)
    after_LN = model.residual_after_LN(layer_idx).squeeze(0)

    logger.info("diff after LN = %.6f", torch.norm(get_back - after_LN).item())


    # --- Extract matrices W_Q, W_K and b_Q, b_K ---
    W_Q_all_heads = model._model.blocks[layer_idx].attn.W_Q 
    W_K_all_heads = model._model.blocks[layer_idx].attn.W_K
    W_V_all_heads = model._model.blocks[layer_idx].attn.W_V

    b_Q_all_heads = model._model.blocks[layer_idx].attn.b_Q 
    b_K_all_heads = model._model.blocks[layer_idx].attn.b_K
    b_V_all_heads = model._model.blocks[layer_idx].attn.b_V

    # ---- Draw projection heatmaps for each head ----
    # v1 = without error
    # v2 = with error from topk
    # v3 = with error from sae out

    prefix = "v3"
    n_heads = model._model.cfg.n_heads
    for head_idx in range(n_heads):
        W_Q_head = W_Q_all_heads[head_idx]
        W_K_head = W_K_all_heads[head_idx]
        b_Q_head = b_Q_all_heads[head_idx]
        b_K_head = b_K_all_heads[head_idx]
        
        plot_projected_contributions_heatmap_per_head(
            sae_decompose_LN[:,-1,:],
            sae_decompose_LN[:,k,:],
            W_Q_head,
            W_K_head,
            b_Q_head,
            b_K_head,
            head_idx,
            k_top,
            len(tokens_str) - 1,
            k,
            filename=f"att_exp/attentions/sae_interaction_{layer_idx}_{head_idx}_k{k}_scale{scale_k}_vs_last{prefix}.png",
            scale = scale_k,
            top_indices=top_indices,
            layer_idx=layer_idx,
            show_values=True
        )


    def patch_resid_pre(tensor, hook):
        #tensor: [batch, pos, d_model]
        out = tensor.clone()
        out[0, k,   :] = sae_decompose.sum(dim = 0)[k, :]
        #out[0, -1,  :] = new_last

        return out

    store = {}

    #---- clean run ----
    _ = model._model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (f"blocks.{layer_idx}.attn.hook_attn_scores", save_activation(store, "scores_before")),
            (f"blocks.{layer_idx}.attn.hook_pattern",     save_activation(store, "pattern_before")),
            (f"blocks.{layer_idx}.hook_resid_mid",        save_activation(store, "resid_mid_before")),
        ],
    )

    #---- run: AFTER patch ----
    _ = model._model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (f"blocks.{layer_idx}.hook_resid_pre",        patch_resid_pre),
            (f"blocks.{layer_idx}.attn.hook_attn_scores", save_scores(store, "scores_after")),
            (f"blocks.{layer_idx}.attn.hook_pattern",     save_pattern(store, "pattern_after")),
            (f"blocks.{layer_idx}.hook_resid_mid",        save_activation(store, "resid_mid_after"))
        ],
    )

    #last_residual_after_attn = model.residual_after_attn(layer=layer_idx)[0, -1, :]
    #print("Last residual after attn:", last_residual_after_attn[:5])

    # ---- extract attention of last token ----
    scores_before  = store["scores_before"][0]     # [head, q_pos, k_pos]
    pattern_before = store["pattern_before"][0]
    scores_after   = store["scores_after"][0]
    pattern_after   = store["pattern_after"][0]


    V_sae_contributions, V_sum = decompose_values_per_head_simple(sae_decompose_LN, W_V_all_heads, b_V_all_heads, pattern_after[:,-1,:])



    index_head = 5
    print("scores before", pattern_before[index_head, -1, :])
    print("scores after", pattern_after[index_head, -1, :])



    # ---- Draw attention score heatmaps ----
    plot_attention_heatmap(
        pattern_before[:,-1,:], 
        tokens_str, 
        title=f"Attention Scores for Last Token (Before Patch) - Layer {layer_idx}",
        filename=f"att_exp/attentions/scores_before_layer_{layer_idx}_{prefix}.png",
        show_values=True
    )

    plot_attention_heatmap(
        pattern_after[:,-1,:], 
        tokens_str, 
        title=f"Attention Scores for Last Token (After Patch) - Layer {layer_idx}",
        filename=f"att_exp/attentions/scores_after_layer_{layer_idx}_{prefix}.png",
        show_values =True
    )

    # ---- Draw heatmap for pattern DIFFERENCE ----
    pattern_diff = pattern_after[:,-1,:] - pattern_before[:,-1,:]
    plot_attention_heatmap(
        pattern_diff,
        tokens_str,
        title=f"Difference in Attention Pattern (After - Before) - Layer {layer_idx}",
        filename=f"att_exp/attentions/pattern_diff_layer_{layer_idx}_{prefix}.png",
        is_diff=True  # Enable difference display mode
    )



    #pattern_after  = store["pattern_after"][0]




    #print("scores before", scores_before[:, -1, 4])

    #base_error_k = errors[k,:]
    #base_error_last = errors[-1,:]
    
    #print(f"\nBase reconstruction error norm for token k ({k}): {torch.norm(base_error_k).item():.4f}")
    #print(f"Base reconstruction error norm for last token: {torch.norm(base_error_last).item():.4f}\n")
    
    last_residual_after_attn = model.residual_after_attn(layer=layer_idx)[0, -1, :]
    print("Last residual after attn:", last_residual_after_attn[:5])

   

    release = "gpt2-small-resid-mid-v5-32k"                              
    sae_id = f"blocks.{layer_idx}.hook_resid_mid"
    sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release, sae_id=sae_id, device=device
    )

    resid_mid_before = store["resid_mid_before"]
    resid_mid_after = store["resid_mid_after"]

    _, sae_cache_before = sae.run_with_cache(resid_mid_before)
    _, sae_cache_after = sae.run_with_cache(resid_mid_after)
    
    acts_post_before  = sae_cache_before["hook_sae_acts_post"]
    acts_post_after = sae_cache_after["hook_sae_acts_post"]

    # ---- Analysis of SAE features from cache (after nonlinearity) ----
    before_indices = analyze_sae_features(acts_post_before[0, -1, :], 100, "resid_mid_before (last token)")
    after_indices = analyze_sae_features(acts_post_after[0, -1, :], 100, "resid_mid_after (last token)")
    after_indices_5 = analyze_sae_features(acts_post_after[0, 5, :], 100, "resid_mid_after (token 5)", compare_indices=after_indices)

    
if __name__ == "__main__":
    main() 