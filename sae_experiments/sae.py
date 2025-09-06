"""
Is that true that hiddens that contain more information have lower avaraged scalar product between SAE features? No.
"""


import torch
from hooked import TransformerLensTransparentLlm
from sae_lens import SAE, HookedSAETransformer
import matplotlib.pyplot as plt
import numpy as np
import os


def get_sae_decomposition(sae, sae_cache_pre, k_top):

    acts_post = sae_cache_pre["hook_sae_acts_post"] # [n_tokens, d_sae]
    top_vals, top_idx = torch.topk(acts_post, k=k_top, dim=-1)
    topk_sae_features = sae.W_dec[top_idx]


    topk_sae_features = topk_sae_features.permute(1, 0, 2).contiguous()  # [top_pos, token_pos, d_model]

    return topk_sae_features

def main():

    MODEL_NAME = "gpt2-small"
    SENTENCE = "The final correct answer: Mark, Max and Smith are in the empty dark room. Smith left. Mark gave flashlight to"
    layer_idx = 11
    device = "cuda"
    dtype = torch.float32
    k_top = 3

    model = TransformerLensTransparentLlm(
        model_name=MODEL_NAME,
        device=device,
        dtype=dtype,
    )
    model.run([SENTENCE])

    tokens_tensor = model._last_run.tokens[0]
    tokens_str = model.tokens_to_strings(tokens_tensor)

    release = "gpt2-small-resid-post-v5-32k"
    sae_id  = f"blocks.{layer_idx-1}.hook_resid_post"
    sae, cfg, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release, sae_id=sae_id, device=device
    )

    resid_pre_all = model.residual_in(layer=layer_idx)[0].to(device)
    _, sae_cache_pre = sae.run_with_cache(resid_pre_all)

    topk_sae_features = get_sae_decomposition(sae, sae_cache_pre, k_top)

    # --- Сalculate and output the average scalar product for the top 10 features of each token 
    n_tokens = topk_sae_features.shape[1]
    avg_similarities = []
    print("\n--- Average Cosine Similarity of Top Features per Token ---")
    for token_idx in range(n_tokens):
        token_features = topk_sae_features[:, token_idx, :]

        normalized_features = token_features / (torch.norm(token_features, dim=1, keepdim=True) + 1e-8)
        cosine_matrix = torch.mm(normalized_features, normalized_features.T)

        lower_triangle_mask = torch.tril(torch.ones(k_top, k_top, device=cosine_matrix.device), diagonal=-1).bool()
        lower_triangle_values = cosine_matrix[lower_triangle_mask]

        avg_similarity = lower_triangle_values.mean().item()
        avg_similarities.append(avg_similarity)

        print(f"Token {token_idx:2d} ({tokens_str[token_idx]:<15}): {avg_similarity:.4f}")
    print("-" * 60)
    # ---

    
    output_dir = "charts"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "sae_feature_similarity.png")

    fig, ax = plt.subplots(figsize=(max(10, n_tokens / 2), 6))
    
    ax.bar(np.arange(n_tokens), avg_similarities, color='skyblue')
    
    ax.set_xticks(np.arange(n_tokens))
    ax.set_xticklabels([f"{i}: {tok}" for i, tok in enumerate(tokens_str)], rotation=45, ha='right')
    
    ax.set_ylabel("Average Cosine Similarity")
    ax.set_xlabel("Token Position and String")
    ax.set_title(f"Avg. Pairwise Cosine Similarity of Top {k_top} SAE Features\nLayer {layer_idx-1} resid_post")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to {filename}")
    # ---


if __name__ == "__main__":
    main()


