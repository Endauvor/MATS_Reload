"""
Find per SAE feature impact when deleting it in the resid_post k-1 and checking resid_mid in the -1 position.
What measured: 
--- KL divergency
--- The decgree of chaos by Ulam coefficient in changing SAE order.
--- Upvote reward for upvoting SAE features in representations. 
--- Newcomer reward for written features into hidden state represantation. 

Creates "knots heatmap" with 4 values in each cell for each layer.
Saves results to att_exp/permutations/ or to att_exp/sae_adds depending on mode.
If you want analyze only KL divergence then look by key word "sae_adds", otherwise look for "permutations". 
"""


import torch
from hooked import TransformerLensTransparentLlm
from contribute import get_attention_contributions
from sae_lens import SAE, HookedSAETransformer
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import torch.nn.functional as F
import numpy as np
from att_exp.permutations_analysis import analysis_results



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

def analyze_sae_features(features: torch.Tensor, top_k: int, description: str):
    """Анализирует и печатает топ-K самых активных SAE фич из готового тензора активаций."""
    top_values, top_indices = torch.topk(features, k=top_k)
    
    print(f"\n--- Top {top_k} SAE Features for: {description} ---")
    print("Feature Index | Activation Value")
    print("----------------|-----------------")
    for val, idx in zip(top_values, top_indices):
        if val.item() > 1e-4: # Печатаем только те, что действительно активны
            print(f"{idx.item():<15} | {val.item():.4f}")

def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor, dim: int = -1) -> torch.Tensor:

    p_log = F.log_softmax(p_logits, dim=dim)  # log p
    q_log = F.log_softmax(q_logits, dim=dim)  # log q
    p = p_log.exp()
    return torch.sum(p * (p_log - q_log), dim=dim)



def get_sae_decomposition(sae, sae_cache_pre, k_top, scale_k, k, errors, sae_index):

    acts_post = sae_cache_pre["hook_sae_acts_post"] # [n_tokens, d_sae]
    top_vals, top_idx = torch.topk(acts_post, k=k_top, dim=-1)
    topk_sae_features = sae.W_dec[top_idx]


    topk_sae_features = topk_sae_features.permute(1, 0, 2).contiguous()  # [top_pos, token_pos, d_model]
    topk_sae_activations = top_vals.permute(1, 0).contiguous()           # [top_pos, token_pos]

    topk_sae_features[sae_index, k, :] = topk_sae_features[sae_index, k, :] * scale_k

    sae_decompose = (topk_sae_activations.unsqueeze(-1) * topk_sae_features \
        + sae.b_dec / k_top) * sae.ln_std + sae.ln_mu / k_top + errors / k_top  

    return sae_decompose




def compute_kl_for_scale(scale_k: float, model, tokens, layer_idx: int, k: int, sae_index: int, k_top: int, device: str) -> tuple[float, float]:
    """
    Вычисляет KL1 и KL2 для заданного scale_k
    """
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
    
    sae_decompose = get_sae_decomposition(sae, sae_cache_pre, k_top, scale_k, k, errors, sae_index)

    sae_decompose_k = sae_decompose[:, k, :]
    get_back_k = sae_decompose_k.sum(dim = 0)

    _, sae_cache_before = sae.run_with_cache(resid_pre_all[k, :])
    _, sae_cache_after = sae.run_with_cache(get_back_k)

    acts_pre_before  = sae_cache_before["hook_sae_acts_pre"]
    acts_pre_after = sae_cache_after["hook_sae_acts_pre"]

    KL1 = kl_from_logits(acts_pre_before, acts_pre_after)

    def patch_resid_pre(tensor, hook):
        #tensor: [batch, pos, d_model]
        out = tensor.clone()
        out[0, k, :] = sae_decompose.sum(dim = 0)[k, :]
        return out

    store = {}

    #---- чистый прогон ----
    _ = model._model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (f"blocks.{layer_idx}.hook_resid_mid", save_activation(store, "resid_mid_before")),
        ],
    )

    #---- прогон: ПОСЛЕ правки ----
    _ = model._model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (f"blocks.{layer_idx}.hook_resid_pre", patch_resid_pre),
            (f"blocks.{layer_idx}.hook_resid_mid", save_activation(store, "resid_mid_after"))
        ],
    )

    release = "gpt2-small-resid-mid-v5-32k"                              
    sae_id = f"blocks.{layer_idx}.hook_resid_mid"
    sae_resid_mid, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release, sae_id=sae_id, device=device
    )

    resid_mid_before = store["resid_mid_before"]
    resid_mid_after = store["resid_mid_after"]

    _, sae_cache_before = sae_resid_mid.run_with_cache(resid_mid_before)
    _, sae_cache_after = sae_resid_mid.run_with_cache(resid_mid_after)
    
    acts_pre_before  = sae_cache_before["hook_sae_acts_pre"]
    acts_pre_after = sae_cache_after["hook_sae_acts_pre"]

    acts_post_before  = sae_cache_before["hook_sae_acts_post"][0,-1,:]
    acts_post_after = sae_cache_after["hook_sae_acts_post"][0,-1,:]

    _, top_indices_first = torch.topk(acts_post_before, k=k_top)
    _, top_indices_second = torch.topk(acts_post_after, k=k_top)
    
    # Convert to lists
    top_indices_first = top_indices_first.tolist()
    top_indices_second = top_indices_second.tolist()

    print(top_indices_first)
    print("\n")
    print(top_indices_second)
    
    upvote, ulam_result, new_reward = analysis_results(top_indices_second, top_indices_first)

    

    KL2 = kl_from_logits(acts_pre_before[0, -1, :], acts_pre_after[0, -1, :])
    
    return top_indices_first, top_indices_second, upvote, ulam_result, new_reward, KL2.item()

def create_kl2_heatmap(kl2_matrix: np.ndarray, tokens_str: list, sae_indices: list, 
                       layer_idx: int, scale_k: float, filename: str):
    """
    Создает и сохраняет хитмапу KL2 значений
    """
    # Создаем хитмапу
    plt.figure(figsize=(max(10, len(tokens_str)), 8))
    
    # Создаем хитмапу с цветовой схемой
    im = plt.imshow(kl2_matrix, cmap='viridis', aspect='auto')
    
    # Добавляем числовые значения в ячейки
    for i in range(len(sae_indices)):
        for j in range(len(tokens_str)):
            value = kl2_matrix[i, j]
            # Выбираем цвет текста в зависимости от яркости фона
            text_color = 'white' if value > kl2_matrix.max() * 0.5 else 'black'
            plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                    color=text_color, fontsize=8, weight='bold')
    
    # Настройка осей
    plt.xticks(range(len(tokens_str)), [f'{i}\n{repr(token)}' for i, token in enumerate(tokens_str)], 
               rotation=45, ha='right')
    plt.yticks(range(len(sae_indices)), [f'SAE {i}' for i in sae_indices])
    
    plt.xlabel('Token Index')
    plt.ylabel('SAE Index')
    plt.title(f'KL2 Heatmap (scale_k={scale_k})\nLayer {layer_idx}')
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar(im)
    cbar.set_label('KL2 Divergence')
    
    # Сохраняем хитмапу
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Диапазон KL2: {kl2_matrix.min():.4f} - {kl2_matrix.max():.4f}")


def create_knots_heatmap(upvote_matrix: np.ndarray, ulam_matrix: np.ndarray, 
                        reward_matrix: np.ndarray, kl2_matrix: np.ndarray,
                        tokens_str: list, sae_indices: list, 
                        layer_idx: int, scale_k: float, filename: str):
    """
    Создает хитмапу с четырьмя метриками в каждой ячейке:
    - upvote (верхний левый)
    - ulam (верхний правый) 
    - reward (нижний левый)
    - kl2 (нижний правый)
    """
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    import matplotlib.cm as cm
    
    # Размеры фигуры
    cell_width = 1.0
    cell_height = 1.0
    fig_width = max(12, len(tokens_str) * 1.5)
    fig_height = max(8, len(sae_indices) * 1.2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
   
    # 1) Чистая синяя -> красная без белой середины:
    cmap = LinearSegmentedColormap.from_list("blue_red", ["#0000FF", "#FF0000"])
    # Если хочешь белую середину – возьми вместо этого:
    # cmap = plt.colormaps["bwr"]  # или plt.colormaps["coolwarm"]

    # 2) Исправь нормализации
    norm_upvote = Normalize(vmin=0, vmax=3.0)   # было 2.0
    norm_ulam   = Normalize(vmin=0, vmax=1.0)
    norm_reward = Normalize(vmin=0, vmax=1.0)
    norm_kl2    = Normalize(vmin=0, vmax=1.0)

    # 3) Защита от NaN/inf (до цикла рисования)
    upvote_matrix = np.nan_to_num(upvote_matrix, nan=0.0, posinf=3.0, neginf=0.0)
    ulam_matrix   = np.nan_to_num(ulam_matrix,   nan=0.0, posinf=1.0, neginf=0.0)
    reward_matrix = np.nan_to_num(reward_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    kl2_matrix    = np.nan_to_num(kl2_matrix,    nan=0.0, posinf=1.0, neginf=0.0)

    # Единая цветовая схема 'plasma'
    cmap = cm.get_cmap('plasma')
    
    # Отрисовка каждой ячейки
    for i in range(len(sae_indices)):
        for j in range(len(tokens_str)):
            # Координаты основной ячейки
            x = j
            y = len(sae_indices) - 1 - i  # Инвертируем Y для правильного отображения
            
            # Размеры подячеек
            sub_width = cell_width / 2
            sub_height = cell_height / 2
            
            # Верхний левый - upvote
            color_upvote = cmap(norm_upvote(upvote_matrix[i, j]))
            rect_upvote = Rectangle((x, y + sub_height), sub_width, sub_height, 
                                  facecolor=color_upvote, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect_upvote)
            
            # Верхний правый - ulam  
            color_ulam = cmap(norm_ulam(ulam_matrix[i, j]))
            rect_ulam = Rectangle((x + sub_width, y + sub_height), sub_width, sub_height,
                                facecolor=color_ulam, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect_ulam)
            
            # Нижний левый - reward
            color_reward = cmap(norm_reward(reward_matrix[i, j]))
            rect_reward = Rectangle((x, y), sub_width, sub_height,
                                  facecolor=color_reward, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect_reward)
            
            # Нижний правый - kl2
            color_kl2 = cmap(norm_kl2(kl2_matrix[i, j]))
            rect_kl2 = Rectangle((x + sub_width, y), sub_width, sub_height,
                                facecolor=color_kl2, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect_kl2)
            
            # Добавляем значения в подячейки (увеличенным шрифтом)
            font_size = min(10, max(8, 80 / max(len(tokens_str), len(sae_indices))))
            
            # Upvote (верхний левый)
            ax.text(x + sub_width/2, y + sub_height + sub_height/2, 
                   f'{upvote_matrix[i, j]:.2f}', 
                   ha='center', va='center', fontsize=font_size, weight='bold', color='black')
            
            # Ulam (верхний правый)
            ax.text(x + sub_width + sub_width/2, y + sub_height + sub_height/2,
                   f'{ulam_matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=font_size, weight='bold', color='black')
            
            # Reward (нижний левый)
            ax.text(x + sub_width/2, y + sub_height/2,
                   f'{reward_matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=font_size, weight='bold', color='black')
            
            # KL2 (нижний правый)
            ax.text(x + sub_width + sub_width/2, y + sub_height/2,
                   f'{kl2_matrix[i, j]:.3f}',
                   ha='center', va='center', fontsize=font_size, weight='bold', color='black')
    
    # Добавляем толстые границы для больших ячеек
    for i in range(len(sae_indices)):
        for j in range(len(tokens_str)):
            x = j
            y = len(sae_indices) - 1 - i
            # Рисуем толстую белую рамку вокруг каждой большой ячейки
            big_cell_rect = Rectangle((x, y), cell_width, cell_height, 
                                    facecolor='none', edgecolor='white', linewidth=3)
            ax.add_patch(big_cell_rect)
    
    # Настройка осей - увеличиваем границы для показа всех ячеек
    ax.set_xlim(-0.1, len(tokens_str) + 0.1)
    ax.set_ylim(-0.1, len(sae_indices) + 0.1)
    
    # Подписи осей
    ax.set_xticks(np.arange(len(tokens_str)) + 0.5)
    ax.set_xticklabels([f'{i}\n{repr(token)}' for i, token in enumerate(tokens_str)], 
                       rotation=45, ha='right')
    
    ax.set_yticks(np.arange(len(sae_indices)) + 0.5)
    ax.set_yticklabels([f'SAE {sae_indices[len(sae_indices)-1-i]}' for i in range(len(sae_indices))])
    
    ax.set_xlabel('Token Index')
    ax.set_ylabel('SAE Index')
    ax.set_title(f'Knots Analysis Heatmap (scale_k={scale_k})\nLayer {layer_idx}\n'
                f'TL:Upvote TR:Ulam BL:Reward BR:KL2')
    
    # Убираем верхние и правые тики
    ax.tick_params(top=False, right=False)
    ax.tick_params(labeltop=False, labelright=False)
    
    # Убираем колорбар - используем только цветовое кодирование
    
    # Сохраняем хитмапу
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Knots heatmap saved: {filename}")
    print(f"  Upvote range: {upvote_matrix.min():.4f} - {upvote_matrix.max():.4f}")
    print(f"  Ulam range: {ulam_matrix.min():.4f} - {ulam_matrix.max():.4f}")
    print(f"  Reward range: {reward_matrix.min():.4f} - {reward_matrix.max():.4f}")
    print(f"  KL2 range: {kl2_matrix.min():.4f} - {kl2_matrix.max():.4f}")


def plot_max_kl2_vs_layers(layers: list, max_kl2_per_layer: list, scale_k: float, suffix: str = ""):
    """
    Строит и сохраняет график зависимости максимального KL2 от номера слоя
    
    Args:
        layers: список номеров слоев
        max_kl2_per_layer: список максимальных KL2 значений для каждого слоя
        scale_k: значение scale_k для подписи
        suffix: суффикс для имени файла
    """
    plt.figure(figsize=(10, 6))
    plt.plot(layers, max_kl2_per_layer, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Layer Index')
    plt.ylabel('Max KL2 Divergence')
    plt.title(f'Maximum KL2 Divergence vs Layer Index (scale_k={scale_k})')
    plt.grid(True, alpha=0.3)
    
    # Добавляем значения на точки
    for i, (layer, max_kl2) in enumerate(zip(layers, max_kl2_per_layer)):
        plt.annotate(f'{max_kl2:.3f}', (layer, max_kl2), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Сохраняем график
    suffix_str = f"_{suffix}" if suffix else ""
    max_kl2_filename = f"att_exp/sae_adds/max_kl2_vs_layer_scale{scale_k}{suffix_str}_x2.png"
    os.makedirs(os.path.dirname(max_kl2_filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(max_kl2_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nГрафик максимальных KL2 сохранен как: {max_kl2_filename}")
    return max_kl2_filename


def main():

    MODEL_NAME = "gpt2-small"
    SENTENCE = "The final correct answer: Mark, Max and Smith are in the empty dark room. Smith left. Mark gave flashlight to"

    layer_idx = 4           
    k = 0             
    device = "cuda"
    dtype = torch.float32   
    k_top = 7
    sae_index = 0
    
    # Создаем лог файл
    log_filename = "att_exp/permutations/high_reward_log.txt"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, 'w') as log_file:
        log_file.write("SAE4 High Reward Analysis Log (new_reward > 0.01)\n")
        log_file.write("=" * 60 + "\n\n")      

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

    
    scale_k = 0  
    num_tokens = len(tokens_str)
    sae_indices = list(range(k_top))  
    layers = list(range(1,12))  
    max_kl2_per_layer = []  
    
    q = 0
    k_top_a = 20

    for layer_idx in layers:
        print(f"\n=== Обрабатываем слой {layer_idx} ===")
        
        kl2_matrix = np.zeros((len(sae_indices), num_tokens-1))
        upvote_matrix = np.zeros((len(sae_indices), num_tokens-1))
        ulam_matrix = np.zeros((len(sae_indices), num_tokens-1))
        reward_matrix = np.zeros((len(sae_indices), num_tokens-1))
        
        for i, sae_idx in enumerate(sae_indices):
            for j, token_idx in enumerate(range(q, num_tokens-1)):
                print(f"layer = {layer_idx}, sae_index = {sae_idx}, token = {token_idx}")
                try:
                    top_indices_first, top_indices_second, upvote, ulam_result, new_reward, kl2 = compute_kl_for_scale(scale_k, model, tokens, layer_idx, token_idx, sae_idx, k_top_a, device)
                    kl2_matrix[i, j] = kl2
                    upvote_matrix[i, j] = upvote  
                    ulam_matrix[i, j] = 1-ulam_result  
                    reward_matrix[i, j] = new_reward
                    
                    print(f"  KL2: {kl2:.4f}, Upvote: {upvote_matrix[i, j]:.3f}, Ulam: {ulam_matrix[i, j]:.3f}, Reward: {reward_matrix[i, j]:.2f}")
                    
                    # Логирование для случаев с высоким new_reward
                    if new_reward > 0.01:
                        with open(log_filename, 'a') as log_file:
                            log_file.write(f"HIGH REWARD FOUND!\n")
                            log_file.write(f"Layer: {layer_idx}, SAE Index: {sae_idx}, Token Index: {token_idx}\n")
                            log_file.write(f"Token: {repr(tokens_str[token_idx])}\n")
                            log_file.write(f"Scale_k: {scale_k}\n")
                            log_file.write(f"Parameters: Upvote={upvote:.4f}, Ulam={1-ulam_result:.4f}, Reward={new_reward:.4f}, KL2={kl2:.4f}\n")
                            log_file.write(f"Top indices before: {top_indices_first}\n")
                            log_file.write(f"Top indices after: {top_indices_second}\n")
                            log_file.write("-" * 50 + "\n\n")
                except Exception as e:
                    print(f"  Ошибка: {e}")
                    kl2_matrix[i, j] = 0
                    upvote_matrix[i, j] = 0
                    ulam_matrix[i, j] = 0 
                    reward_matrix[i, j] = 0
        
        # Записываем максимальное значение KL2 для текущего слоя
        max_kl2_layer = np.max(kl2_matrix)
        max_kl2_per_layer.append(max_kl2_layer)
        print(f"Максимальное KL2 для слоя {layer_idx}: {max_kl2_layer:.4f}")
        
        # Создаем обычную KL2 хитмапу
        filename = f"att_exp/sae_adds/kl2_heatmap_layer{layer_idx}_scale{scale_k}.png"
        create_kl2_heatmap(kl2_matrix, tokens_str[q:-1], sae_indices, layer_idx, scale_k, filename)
        
        # Создаем новую knots хитмапу с четырьмя метриками
        knots_filename = f"att_exp/permutations/_knots_heatmap_layer{layer_idx}_scale{scale_k}.png"
        create_knots_heatmap(upvote_matrix, ulam_matrix, reward_matrix, kl2_matrix,
                           tokens_str[q:-1], sae_indices, layer_idx, scale_k, knots_filename)
    
    # Строим график зависимости максимального KL2 от номера слоя
    plot_max_kl2_vs_layers(layers, max_kl2_per_layer, scale_k)
    
    print("Максимальные KL2 по слоям:")
    for layer, max_kl2 in zip(layers, max_kl2_per_layer):
        print(f"  Слой {layer}: {max_kl2:.4f}")
    
    # Финальная запись в лог
    with open(log_filename, 'a') as log_file:
        log_file.write("=" * 60 + "\n")
        log_file.write("ANALYSIS COMPLETED\n")
        log_file.write(f"Processed layers: {layers}\n")
        log_file.write(f"Scale_k: {scale_k}\n")
        log_file.write(f"Threshold for logging: new_reward > 0.01\n")
        log_file.write("=" * 60 + "\n")
    
    print(f"High reward log saved to: {log_filename}")

    


if __name__ == "__main__":
    main() 