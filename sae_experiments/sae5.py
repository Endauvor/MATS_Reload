"""
Find per token impact when deleting it in attention output and checking resid_mid in the -1 position.
Heatmap "knots heatmap" is created with 4 values in each cell: 
--- KL divergency
--- The decgree of chaos by Ulam coefficient in changing SAE order.
--- Upvote reward for upvoting SAE features in representations. 
--- Newcomer reward for written features into hidden state represantation. 

Based on sae4.py, but per token analysis. 
Overall 2-side heatmap is created. 
On the left: if we delete the whole attention layer. 
On the right: if we delete a token add after attention.
You can see that this is not true that if a layer is "important", then some token is. 
"""

from att_exp.permutations_analysis import analysis_results
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from hooked import TransformerLensTransparentLlm
from contribute import get_attention_contributions
import torch
from sae_lens import SAE, HookedSAETransformer
import os
from datetime import datetime
import requests

import torch.nn.functional as F


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p_log = F.log_softmax(p_logits, dim=dim)  # log p
    q_log = F.log_softmax(q_logits, dim=dim)  # log q
    p = p_log.exp()
    return torch.sum(p * (p_log - q_log), dim=dim)  

def entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=dim)
    p = log_p.exp()
    return -(p * log_p).sum(dim=dim)


def plot_head_kl_heatmaps(head_kl_matrix: torch.Tensor, tokens_str: list, n_heads: int = 12):
    """
    Строит и сохраняет хитмапы KL дивергенций для каждой головы внимания
    
    Args:
        head_kl_matrix: torch тензор размера [n_heads, n_layers, n_tokens] с KL дивергенциями
        tokens_str: список токенов
        n_heads: количество голов внимания
    """
    os.makedirs("att_exp/token_ads", exist_ok=True)
    
    # Конвертируем в numpy для matplotlib
    head_kl_numpy = head_kl_matrix.detach().cpu().numpy()
    n_layers = head_kl_numpy.shape[1]
    
    for head_idx in range(n_heads):
        plt.figure(figsize=(max(10, len(tokens_str)), 8))
        
        # Извлекаем данные для конкретной головы
        head_data = head_kl_numpy[head_idx, :, :]  # [n_layers, n_tokens]
        
        # Создаем хитмапу (origin='lower' чтобы слои шли снизу вверх)
        im = plt.imshow(head_data, cmap='viridis', aspect='auto', origin='lower')
        
        # Добавляем числовые значения в ячейки
        for i in range(n_layers):
            for j in range(len(tokens_str)):
                value = head_data[i, j]
                if not np.isnan(value):  # Проверяем на NaN значения
                    # Выбираем цвет текста в зависимости от яркости фона
                    text_color = 'white' if value > np.nanmax(head_data) * 0.5 else 'black'
                    plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                            color=text_color, fontsize=8, weight='bold')
        
        # Настройка осей
        plt.xticks(range(len(tokens_str)), [f'{i}\n{repr(token)}' for i, token in enumerate(tokens_str)], 
                   rotation=45, ha='right')
        plt.yticks(range(n_layers), [f'Layer {i}' for i in range(n_layers)])
        
        plt.xlabel('Token Index')
        plt.ylabel('Layer Index')
        plt.title(f'KL Divergence Heatmap for Attention Head {head_idx + 1}')
        
        # Добавляем цветовую шкалу
        cbar = plt.colorbar(im)
        cbar.set_label('KL Divergence')
        
        # Сохраняем хитмапу
        filename = f"att_exp/token_ads/head_{head_idx + 1}_kl_heatmap.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Хитмапа для головы {head_idx + 1} сохранена: {filename}")


def plot_layer_head_token_heatmaps(layer_head_token_matrices: list, tokens_str: list, n_layers: int = 10):
    """
    Строит и сохраняет хитмапы KL дивергенций для каждого слоя
    где по Y идут головы (1-12), а по X токены
    
    Args:
        layer_head_token_matrices: список torch тензоров размера [n_heads, n_tokens] для каждого слоя
        tokens_str: список токенов
        n_layers: количество слоев
    """
    os.makedirs("att_exp/token_ads", exist_ok=True)
    
    for layer_idx, head_token_matrix in enumerate(layer_head_token_matrices):
        # Конвертируем в numpy для matplotlib
        head_token_numpy = head_token_matrix.detach().cpu().numpy()
        n_heads, n_tokens = head_token_numpy.shape
        
        plt.figure(figsize=(max(10, len(tokens_str)), max(8, n_heads)))
        
        # Создаем хитмапу
        im = plt.imshow(head_token_numpy, cmap='viridis', aspect='auto')
        
        # Добавляем числовые значения в ячейки
        for i in range(n_heads):
            for j in range(n_tokens):
                value = head_token_numpy[i, j]
                if not np.isnan(value):  # Проверяем на NaN значения
                    # Выбираем цвет текста в зависимости от яркости фона
                    text_color = 'white' if value > np.nanmax(head_token_numpy) * 0.5 else 'black'
                    plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                            color=text_color, fontsize=8, weight='bold')
        
        # Настройка осей
        plt.xticks(range(len(tokens_str)), [f'{i}\n{repr(token)}' for i, token in enumerate(tokens_str)], 
                   rotation=45, ha='right')
        plt.yticks(range(n_heads), [f'Head {i+1}' for i in range(n_heads)])
        
        plt.xlabel('Token Index')
        plt.ylabel('Attention Head')
        plt.title(f'KL Divergence Heatmap for Layer {layer_idx} (Heads vs Tokens)')
        
        # Добавляем цветовую шкалу
        cbar = plt.colorbar(im)
        cbar.set_label('KL Divergence')
        
        # Сохраняем хитмапу
        filename = f"att_exp/token_ads/layer_{layer_idx}_heads_tokens_kl_heatmap.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Хитмапа слой {layer_idx} (головы vs токены) сохранена: {filename}")


def plot_layer_token_kl_heatmap(layer_token_kl_matrix: torch.Tensor, tokens_str: list, n_layers: int = 10, kl_initial_values: list = None):
    """
    Строит и сохраняет хитмапу KL дивергенций по слоям и токенам
    где по X токены, по Y слои
    
    Args:
        layer_token_kl_matrix: torch тензор размера [n_layers, n_tokens] с KL дивергенциями
        tokens_str: список токенов
        n_layers: количество слоев
        kl_initial_values: список значений kl_initial_whole для каждого слоя
    """
    os.makedirs("att_exp/token_ads", exist_ok=True)
    
    # Конвертируем в numpy для matplotlib
    layer_token_numpy = layer_token_kl_matrix.detach().cpu().numpy()
    n_tokens = len(tokens_str)
    
    # Создаем фигуру с двумя субплотами
    if kl_initial_values is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n_tokens + 2), max(8, n_layers)), 
                                      gridspec_kw={'width_ratios': [1, n_tokens], 'wspace': 0.1})
        
        # Левая хитмапа для kl_initial_values
        kl_initial_numpy = np.array(kl_initial_values).reshape(-1, 1)  # [n_layers, 1]
        im1 = ax1.imshow(kl_initial_numpy, cmap='viridis', aspect='auto', origin='lower')
        
        # Добавляем числовые значения в ячейки левой хитмапы
        for i in range(n_layers):
            value = kl_initial_numpy[i, 0]
            if not np.isnan(value):
                # Выбираем цвет текста в зависимости от яркости фона
                text_color = 'white' if value > np.nanmax(kl_initial_numpy) * 0.5 else 'black'
                ax1.text(0, i, f'{value:.3f}', ha='center', va='center', 
                        color=text_color, fontsize=8, weight='bold')
        
        # Настройка левой хитмапы
        ax1.set_xticks([0])
        ax1.set_xticklabels(['Initial'])
        ax1.set_yticks(range(n_layers))
        ax1.set_yticklabels([f'Layer {i}' for i in range(n_layers)])
        ax1.set_xlabel('Initial KL')
        ax1.set_ylabel('Layer Index')
        ax1.set_title('KL Initial vs Whole')
        
        # Правая хитмапа для основных данных
        im2 = ax2.imshow(layer_token_numpy, cmap='viridis', aspect='auto', origin='lower')
        
        # Добавляем числовые значения в ячейки правой хитмапы
        for i in range(n_layers):
            for j in range(n_tokens):
                value = layer_token_numpy[i, j]
                if not np.isnan(value):
                    # Выбираем цвет текста в зависимости от яркости фона
                    text_color = 'white' if value > np.nanmax(layer_token_numpy) * 0.5 else 'black'
                    ax2.text(j, i, f'{value:.3f}', ha='center', va='center', 
                            color=text_color, fontsize=8, weight='bold')
        
        # Настройка правой хитмапы
        ax2.set_xticks(range(len(tokens_str)))
        ax2.set_xticklabels([f'{i}\n{repr(token)}' for i, token in enumerate(tokens_str)], 
                           rotation=45, ha='right')
        ax2.set_yticks(range(n_layers))
        ax2.set_yticklabels([f'Layer {i}' for i in range(n_layers)])
        ax2.set_xlabel('Token Index')
        ax2.set_title('KL Divergence: Whole vs Whole-minus-Token-Add')
        
        # Добавляем цветовую шкалу только для правой хитмапы
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('KL Divergence')
        
    else:
        # Если kl_initial_values не предоставлены, рисуем как раньше
        plt.figure(figsize=(max(10, n_tokens), max(8, n_layers)))
        
        # Создаем хитмапу (origin='lower' чтобы слои шли снизу вверх)
        im = plt.imshow(layer_token_numpy, cmap='viridis', aspect='auto', origin='lower')
        
        # Добавляем числовые значения в ячейки
        for i in range(n_layers):
            for j in range(n_tokens):
                value = layer_token_numpy[i, j]
                if not np.isnan(value):  # Проверяем на NaN значения
                    # Выбираем цвет текста в зависимости от яркости фона
                    text_color = 'white' if value > np.nanmax(layer_token_numpy) * 0.5 else 'black'
                    plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                            color=text_color, fontsize=8, weight='bold')
        
        # Настройка осей
        plt.xticks(range(len(tokens_str)), [f'{i}\n{repr(token)}' for i, token in enumerate(tokens_str)], 
                   rotation=45, ha='right')
        plt.yticks(range(n_layers), [f'Layer {i}' for i in range(n_layers)])
        
        plt.xlabel('Token Index')
        plt.ylabel('Layer Index')
        plt.title('KL Divergence: Whole vs Whole-minus-Token-Add (Layers vs Tokens)')
        
        # Добавляем цветовую шкалу
        cbar = plt.colorbar(im)
        cbar.set_label('KL Divergence')
    
    # Сохраняем хитмапу
    filename = "att_exp/token_ads/layer_token_kl_heatmap.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Хитмапа слой-токен сохранена: {filename}")


def kl_top_k_features(acts_pre_for_adds: torch.Tensor, acts_pre_whole: torch.Tensor, K: int = 20) -> torch.Tensor:
    """
    Calculate KL divergence on top-K features from softmax(acts_pre_whole)
    
    Args:
        acts_pre_for_adds: tensor of shape [n_adds, d_sae] - SAE activations for individual additions
        acts_pre_whole: tensor of shape [d_sae] - SAE activations for whole residual
        K: number of top features to consider
        
    Returns:
        kl_add_vs_whole_top: tensor of shape [n_adds] - KL divergence values
    """
    # Get top-K features from whole residual
    whole_prob = F.softmax(acts_pre_whole, dim=-1)               
    top_vals, top_idx = torch.topk(whole_prob, k=K, dim=-1)     

    # Normalize top probabilities
    whole_top_prob = top_vals.softmax(dim = -1)
    whole_top_log = torch.log(whole_top_prob + 1e-12)               

    # Get corresponding features for additions
    adds_top_logits = acts_pre_for_adds[:, top_idx]                  # [n_adds, K]
    adds_top_log = F.log_softmax(adds_top_logits, dim=-1)         # [n_adds, K]
    adds_top_prob = adds_top_log.exp()

    # Calculate KL divergence
    kl_add_vs_whole_top = torch.sum(adds_top_prob * (adds_top_log - whole_top_log), dim=-1)
    
    return kl_add_vs_whole_top


def plot_kl_histogram(kl_values: torch.Tensor, title: str = "KL Divergence Distribution", 
                     filename: str = "information_gain_histogram.png", bins: int = 30):
    """
    Plot and save KL divergence histogram
    
    Args:
        kl_values: tensor with KL divergence values
        title: plot title
        filename: filename for saving
        bins: number of histogram bins
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    kl_numpy = kl_values.detach().cpu().numpy()
    
    # Plot histogram
    plt.hist(kl_numpy, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('KL(add_i || whole)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_kl = np.mean(kl_numpy)
    std_kl = np.std(kl_numpy)
    plt.axvline(mean_kl, color='red', linestyle='--', label=f'Mean: {mean_kl:.4f}')
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print(f"\nHistogram saved to '{filename}'")
    print(f"KL divergence statistics:")
    print(f"  Mean: {mean_kl:.4f}")
    print(f"  Std: {std_kl:.4f}")
    print(f"  Min: {np.min(kl_numpy):.4f}")
    print(f"  Max: {np.max(kl_numpy):.4f}")
    
    return mean_kl, std_kl

def analyze_sae_features(features: torch.Tensor, top_k: int, description: str, layer_idx: int = None, log_file=None):
    """Анализирует и печатает топ-K самых активных SAE фич из готового тензора активаций."""
    top_values, top_indices = torch.topk(features, k=top_k)
    
    output_lines = []
    output_lines.append(f"\n--- Top {top_k} SAE Features for: {description} ---")
    output_lines.append("Feature Index | Activation Value | Top 3 Positive Words")
    output_lines.append("--------------|------------------|---------------------")
    
    for val, idx in zip(top_values, top_indices):
        if val.item() > 1e-4: # Печатаем только те, что действительно активны
            feature_idx = idx.item()
            activation_val = val.item()
            
            
            # Формируем правильное название SAE
            
            sae_name = f"{layer_idx}-res_mid_32k-oai"  # Для остальных слоёв используем новый формат с подчёркиванием
            
            
            url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{sae_name}/{feature_idx}"
            
            # Получаем данные из API
            try:
                response = requests.get(
                    url,
                    headers={
                        "x-api-key": "sk-np-mr5bqLiLmlKbpioVXmplW5m0BkNmtBSEdTsruyS55HA0"
                    },
                    timeout=30  # Увеличиваем таймаут
                )
                
                print(f"Debug: URL = {url}, Status = {response.status_code}")  # Отладочная информация
                
                if response.status_code == 200:
                    data = response.json()
                    pos_str = data.get("pos_str", [])
                    
                    # Берем первые 3 слова из pos_str
                    top3_words = pos_str[:2] if pos_str else ["N/A", "N/A", "N/A"]
                    top3_str = ", ".join(top3_words)
                else:
                    # Показываем больше информации об ошибке
                    try:
                        error_text = response.text[:100]
                        top3_str = f"API Error {response.status_code}: {error_text}"
                    except:
                        top3_str = f"API Error: {response.status_code}"
                    
            except Exception as e:
                top3_str = f"Request Error: {str(e)[:50]}..."
            
            line = f"{feature_idx:<13} | {activation_val:<16.4f} | {top3_str}"
            output_lines.append(line)
    
    # Выводим в консоль
    for line in output_lines:
        print(line)
    
    # Записываем в файл если передан
    if log_file:
        for line in output_lines:
            log_file.write(line + "\n")
        log_file.write("\n")

def compute_token_adds_statistics(token_adds_to_the_last: torch.Tensor) -> tuple:
    """
    Вычисляет статистики для добавок токенов:
    1. Среднее косинусное сходство между нормированными векторами добавок
    2. Среднюю, минимальную и максимальную норму добавок
    
    Args:
        token_adds_to_the_last: tensor of shape [n_tokens, d_model] - добавки токенов
        
    Returns:
        tuple: (mean_cosine_similarity, mean_norm, min_norm, max_norm)
    """
    # Вычисляем нормы добавок
    norms = torch.norm(token_adds_to_the_last, dim=-1)  # [n_tokens]
    mean_norm = norms.mean().item()
    min_norm = norms.min().item() 
    max_norm = norms.max().item()
    
    # Нормализуем векторы добавок
    normalized_adds = torch.nn.functional.normalize(token_adds_to_the_last, dim=-1)  # [n_tokens, d_model]
    
    # Вычисляем матрицу косинусных сходств
    cosine_matrix = torch.mm(normalized_adds, normalized_adds.T)  # [n_tokens, n_tokens]
    
    # Берем только значения под диагональю (нижний треугольник без диагонали)
    n_tokens = cosine_matrix.shape[0]
    lower_triangle_mask = torch.tril(torch.ones(n_tokens, n_tokens), diagonal=-1).bool()
    lower_triangle_values = cosine_matrix[lower_triangle_mask]
    
    # Вычисляем среднее значение
    mean_cosine_similarity = lower_triangle_values.mean().item() if len(lower_triangle_values) > 0 else 0.0
    
    return mean_cosine_similarity, mean_norm, min_norm, max_norm
    
def main():
   
    MODEL_NAME = "gpt2-small"
    SENTENCE = "The final correct answer: Mark, Max and Smith are in the empty dark room. Smith left. Mark gave flashlight to" #. Mark gave flashlight to
    k_top = 20

    log_filename = f"att_exp/permutations2/logs.txt"
    
    # Создаем файл логов
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, 'w') as log_file:
        log_file.write("SAE Feature Analysis Log\n")
        log_file.write("=" * 50 + "\n\n")
   
    model = TransformerLensTransparentLlm(
        model_name=MODEL_NAME, 
        device="cuda",
        dtype=torch.float32
    )

    model.run([SENTENCE])

    # Generate next tokens
    print("Generating next tokens...")
    
    # Get the last token as starting point
    current_tokens = model._last_run.tokens[0].clone()
    generated_tokens = []
    
    for _ in range(5):
        # Get logits for the next token
        with torch.no_grad():
            logits = model._cached_model(current_tokens.unsqueeze(0))  # Add batch dimension
            next_token_logits = logits[0, -1, :]  # Get logits for the last position
        
        # Get the most likely next token (greedy decoding)
        next_token = torch.argmax(next_token_logits).unsqueeze(0)
        
        # Add to sequence
        current_tokens = torch.cat([current_tokens, next_token])
        generated_tokens.append(model._cached_model.tokenizer.decode(next_token.item()))
    
    print(f"Generated tokens: {generated_tokens}")
    
    full_sentence = SENTENCE + "".join(generated_tokens)
    print(f"Full sentence: {full_sentence}")

    tokens_tensor = model._last_run.tokens[0] 
    tokens_str = model.tokens_to_strings(tokens_tensor)
    
    print("\n Tokens: ")
    for i, token in enumerate(tokens_str):
        print(f"{i}: {repr(token)}")

   
    n_layers = 12
    n_tokens = len(tokens_str)
       

    # Подготовка матриц для сбора данных
    q = 0
    layers = list(range(q, n_layers))
    
    # Матрицы для значений по токенам (правая часть хитмапы)
    upvote_t_matrix = np.zeros((len(layers), n_tokens))
    ulam_t_matrix = np.zeros((len(layers), n_tokens))
    reward_t_matrix = np.zeros((len(layers), n_tokens))
    kl2_t_matrix = np.zeros((len(layers), n_tokens))
    
    # Списки для значений по слоям (левая часть хитмапы)
    upvote_l_values = []
    ulam_l_values = []
    reward_l_values = []
    kl2_l_values = []

    for layer_i, layer_idx in enumerate(layers):
        print(f"\n=== Processing layer {layer_idx} ===")
        
        # Вычисляем значения для целого слоя (l-индекс) - один раз на слой
        all_adds_to_the_last = model.decomposed_attn(layer=layer_idx)[-1, :, :, :]
        token_adds_to_the_last = all_adds_to_the_last.sum(dim = 1)
        
        last_residual_before_attn = model.residual_in(layer=layer_idx)[0, -1, :]
        last_residual_after_attn = model.residual_after_attn(layer=layer_idx)[0, -1, :]
        
        device = "cuda"
        release = "gpt2-small-resid-mid-v5-32k"                              
        sae_id = f"blocks.{layer_idx}.hook_resid_mid"  

        sae, cfg, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release=release, sae_id=sae_id, device=device
        )

        _, sae_cache_initial = sae.run_with_cache(last_residual_before_attn)
        acts_pre_initial = sae_cache_initial["hook_sae_acts_pre"]
        acts_post_initial = sae_cache_initial["hook_sae_acts_post"]

        _, sae_cache_whole = sae.run_with_cache(last_residual_after_attn)
        acts_pre_whole = sae_cache_whole["hook_sae_acts_pre"]
        acts_post_after_attn = sae_cache_whole["hook_sae_acts_post"]

        # Вычисляем значения для слоя (l-индекс)
        _, top_indices_without_attn = torch.topk(acts_post_initial, k=k_top)
        _, top_indices_after_attn = torch.topk(acts_post_after_attn, k=k_top)
        
        top_indices_without_attn = top_indices_without_attn.tolist()
        top_indices_after_attn = top_indices_after_attn.tolist()
        
        print("without attn", top_indices_without_attn)
        print("with attn", top_indices_after_attn)

        upvote_l, ulam_result_l, new_reward_l = analysis_results(top_indices_without_attn, top_indices_after_attn)
        KL2_l = (kl_from_logits(acts_pre_whole, acts_pre_initial) + kl_from_logits(acts_pre_initial, acts_pre_whole)) / 2
        
        # Сохраняем значения для слоя
        upvote_l_values.append(upvote_l)
        ulam_l_values.append(1 - ulam_result_l)  # Инвертируем как в sae4.py
        reward_l_values.append(new_reward_l)
        kl2_l_values.append(KL2_l.item() if hasattr(KL2_l, 'item') else KL2_l)
        
        # Логирование для слоя
        with open(log_filename, 'a') as log_file:
            log_file.write(f"LAYER {layer_idx} (L-index)\n")
            log_file.write(f"Without attention: {top_indices_without_attn}\n")
            log_file.write(f"With attention: {top_indices_after_attn}\n")
            log_file.write(f"Upvote_l: {upvote_l:.4f}, Ulam_l: {1-ulam_result_l:.4f}, Reward_l: {new_reward_l:.4f}, KL2_l: {kl2_l_values[-1]:.4f}\n")
            log_file.write("-" * 40 + "\n")
        
        
        for token_idx in range(n_tokens):
            print(f"  Processing token {token_idx}: {repr(tokens_str[token_idx])}")
            
            whole_minus_token_add = last_residual_after_attn - token_adds_to_the_last[token_idx, :]

            _, sae_cache_whole_minus_token_adds = sae.run_with_cache(whole_minus_token_add)
            acts_pre_minus_token_add = sae_cache_whole_minus_token_adds["hook_sae_acts_pre"]
            acts_post_after_attn_minus_token_add = sae_cache_whole_minus_token_adds["hook_sae_acts_post"]

            _, top_indices_after_attn_minus_token_add = torch.topk(acts_post_after_attn_minus_token_add, k=k_top)
            top_indices_after_attn_minus_token_add = top_indices_after_attn_minus_token_add.tolist()

            upvote_t, ulam_result_t, new_reward_t = analysis_results(top_indices_after_attn_minus_token_add, top_indices_after_attn)
            
            KL2_t = (kl_from_logits(acts_pre_whole, acts_pre_minus_token_add) + kl_from_logits(acts_pre_minus_token_add, acts_pre_whole)) / 2

            # Сохраняем значения в матрицы
            upvote_t_matrix[layer_i, token_idx] = upvote_t
            ulam_t_matrix[layer_i, token_idx] = 1 - ulam_result_t  # Инвертируем как в sae4.py
            reward_t_matrix[layer_i, token_idx] = new_reward_t
            kl2_t_matrix[layer_i, token_idx] = KL2_t.item() if hasattr(KL2_t, 'item') else KL2_t
            
            print(f"    Upvote_t: {upvote_t:.3f}, Ulam_t: {1-ulam_result_t:.3f}, Reward_t: {new_reward_t:.3f}, KL2_t: {kl2_t_matrix[layer_i, token_idx]:.4f}")
            
            # Логирование для токена
            with open(log_filename, 'a') as log_file:
                log_file.write(f"  Token {token_idx} ({repr(tokens_str[token_idx])}) (T-index)\n")
                log_file.write(f"  After attention: {top_indices_after_attn}\n")
                log_file.write(f"  After attn minus token add: {top_indices_after_attn_minus_token_add}\n")
                log_file.write(f"  Upvote_t: {upvote_t:.4f}, Ulam_t: {1-ulam_result_t:.4f}, Reward_t: {new_reward_t:.4f}, KL2_t: {kl2_t_matrix[layer_i, token_idx]:.4f}\n")
                log_file.write("  " + "-" * 35 + "\n")

    # Создаем хитмапу

    filename = "att_exp/permutations2/layer_token_knots_heatmap.png"
    create_layer_token_knots_heatmap(
        upvote_l_values, ulam_l_values, reward_l_values, kl2_l_values,
        upvote_t_matrix, ulam_t_matrix, reward_t_matrix, kl2_t_matrix,
        tokens_str, layers, filename
    )

    # Финальная запись в лог
    with open(log_filename, 'a') as log_file:
        log_file.write("\n" + "=" * 50 + "\n")
        log_file.write("ANALYSIS COMPLETED\n")
        log_file.write(f"Heatmap saved to: {filename}\n")
        log_file.write("=" * 50 + "\n")

    print(f"Results saved to: {log_filename}")
    print(f"Heatmap saved to: {filename}")


def create_layer_token_knots_heatmap(upvote_l_values: list, ulam_l_values: list, 
                                   reward_l_values: list, kl2_l_values: list,
                                   upvote_t_matrix: np.ndarray, ulam_t_matrix: np.ndarray, 
                                   reward_t_matrix: np.ndarray, kl2_t_matrix: np.ndarray,
                                   tokens_str: list, layers: list, filename: str):
    """
    Создает хитмапу с двумя частями:
    - Левая: значения по слоям (upvote_l, ulam_l, reward_l, kl2_l) - один столбец
    - Правая: матрица значений по токенам и слоям (upvote_t, ulam_t, reward_t, kl2_t)
    
    Args:
        upvote_l_values: список значений upvote_l для каждого слоя
        ulam_l_values: список значений ulam_l для каждого слоя  
        reward_l_values: список значений reward_l для каждого слоя
        kl2_l_values: список значений kl2_l для каждого слоя
        upvote_t_matrix: матрица upvote_t [n_layers, n_tokens]
        ulam_t_matrix: матрица ulam_t [n_layers, n_tokens]
        reward_t_matrix: матрица reward_t [n_layers, n_tokens]
        kl2_t_matrix: матрица kl2_t [n_layers, n_tokens]
        tokens_str: список токенов
        layers: список номеров слоев
        filename: имя файла для сохранения
    """
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    import matplotlib.cm as cm
    
    # Конвертируем списки в numpy массивы
    upvote_l_array = np.array(upvote_l_values)
    ulam_l_array = np.array(ulam_l_values) 
    reward_l_array = np.array(reward_l_values)
    kl2_l_array = np.array(kl2_l_values)
    
    # Размеры
    n_layers = len(layers)
    n_tokens = len(tokens_str)
    cell_width = 1.0
    cell_height = 1.0
    
    # Размеры фигуры: левая часть (1 столбец) + правая часть (n_tokens столбцов)
    fig_width = max(14, (1 + n_tokens) * 1.5)
    fig_height = max(8, n_layers * 1.2)
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(fig_width, fig_height),
                                           gridspec_kw={'width_ratios': [1, n_tokens], 'wspace': 0.05})
    
    # Цветовая схема
    cmap = cm.get_cmap('plasma')
    
    # Нормализации (объединяем данные из левой и правой частей для единой шкалы)
    all_upvote = np.concatenate([upvote_l_array, upvote_t_matrix.flatten()])
    all_ulam = np.concatenate([ulam_l_array, ulam_t_matrix.flatten()])
    all_reward = np.concatenate([reward_l_array, reward_t_matrix.flatten()])
    all_kl2 = np.concatenate([kl2_l_array, kl2_t_matrix.flatten()])
    
    norm_upvote = Normalize(vmin=np.min(all_upvote), vmax=np.max(all_upvote))
    norm_ulam   = Normalize(vmin=np.min(all_ulam), vmax=np.max(all_ulam))
    norm_reward = Normalize(vmin=np.min(all_reward), vmax=np.max(all_reward))
    norm_kl2    = Normalize(vmin=np.min(all_kl2), vmax=np.max(all_kl2))
    
    # Защита от NaN/inf
    upvote_l_array = np.nan_to_num(upvote_l_array, nan=0.0, posinf=3.0, neginf=0.0)
    ulam_l_array   = np.nan_to_num(ulam_l_array,   nan=0.0, posinf=1.0, neginf=0.0)
    reward_l_array = np.nan_to_num(reward_l_array, nan=0.0, posinf=1.0, neginf=0.0)
    kl2_l_array    = np.nan_to_num(kl2_l_array,    nan=0.0, posinf=1.0, neginf=0.0)
    
    upvote_t_matrix = np.nan_to_num(upvote_t_matrix, nan=0.0, posinf=3.0, neginf=0.0)
    ulam_t_matrix   = np.nan_to_num(ulam_t_matrix,   nan=0.0, posinf=1.0, neginf=0.0)
    reward_t_matrix = np.nan_to_num(reward_t_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    kl2_t_matrix    = np.nan_to_num(kl2_t_matrix,    nan=0.0, posinf=1.0, neginf=0.0)
    
    # ========== ЛЕВАЯ ЧАСТЬ (один столбец для значений по слоям) ==========
    for i in range(n_layers):
        # Координаты ячейки (инвертируем Y для правильного отображения)
        x = 0
        y = n_layers - 1 - i
        
        # Размеры подячеек
        sub_width = cell_width / 2
        sub_height = cell_height / 2
        
        # Верхний левый - upvote_l
        color_upvote = cmap(norm_upvote(upvote_l_array[i]))
        rect_upvote = Rectangle((x, y + sub_height), sub_width, sub_height, 
                              facecolor=color_upvote, edgecolor='white', linewidth=0.5)
        ax_left.add_patch(rect_upvote)
        
        # Верхний правый - ulam_l  
        color_ulam = cmap(norm_ulam(ulam_l_array[i]))
        rect_ulam = Rectangle((x + sub_width, y + sub_height), sub_width, sub_height,
                            facecolor=color_ulam, edgecolor='white', linewidth=0.5)
        ax_left.add_patch(rect_ulam)
        
        # Нижний левый - reward_l
        color_reward = cmap(norm_reward(reward_l_array[i]))
        rect_reward = Rectangle((x, y), sub_width, sub_height,
                              facecolor=color_reward, edgecolor='white', linewidth=0.5)
        ax_left.add_patch(rect_reward)
        
        # Нижний правый - kl2_l
        color_kl2 = cmap(norm_kl2(kl2_l_array[i]))
        rect_kl2 = Rectangle((x + sub_width, y), sub_width, sub_height,
                            facecolor=color_kl2, edgecolor='white', linewidth=0.5)
        ax_left.add_patch(rect_kl2)
        
        # Добавляем значения в подячейки
        font_size = min(10, max(8, 80 / max(1, n_layers)))
        
        # Upvote (верхний левый)
        ax_left.text(x + sub_width/2, y + sub_height + sub_height/2, 
                   f'{upvote_l_array[i]:.2f}', 
                   ha='center', va='center', fontsize=font_size, weight='bold', color='black')
        
        # Ulam (верхний правый)
        ax_left.text(x + sub_width + sub_width/2, y + sub_height + sub_height/2,
                   f'{ulam_l_array[i]:.2f}',
                   ha='center', va='center', fontsize=font_size, weight='bold', color='black')
        
        # Reward (нижний левый)
        ax_left.text(x + sub_width/2, y + sub_height/2,
                   f'{reward_l_array[i]:.2f}',
                   ha='center', va='center', fontsize=font_size, weight='bold', color='black')
        
        # KL2 (нижний правый)
        ax_left.text(x + sub_width + sub_width/2, y + sub_height/2,
                   f'{kl2_l_array[i]:.3f}',
                   ha='center', va='center', fontsize=font_size, weight='bold', color='black')
        
        # Рисуем толстую белую рамку вокруг большой ячейки
        big_cell_rect = Rectangle((x, y), cell_width, cell_height, 
                                facecolor='none', edgecolor='white', linewidth=3)
        ax_left.add_patch(big_cell_rect)
    
    # ========== ПРАВАЯ ЧАСТЬ (матрица по токенам и слоям) ==========
    for i in range(n_layers):
        for j in range(n_tokens):
            # Координаты ячейки
            x = j
            y = n_layers - 1 - i
            
            # Размеры подячеек
            sub_width = cell_width / 2
            sub_height = cell_height / 2
            
            # Верхний левый - upvote_t
            color_upvote = cmap(norm_upvote(upvote_t_matrix[i, j]))
            rect_upvote = Rectangle((x, y + sub_height), sub_width, sub_height, 
                                  facecolor=color_upvote, edgecolor='white', linewidth=0.5)
            ax_right.add_patch(rect_upvote)
            
            # Верхний правый - ulam_t  
            color_ulam = cmap(norm_ulam(ulam_t_matrix[i, j]))
            rect_ulam = Rectangle((x + sub_width, y + sub_height), sub_width, sub_height,
                                facecolor=color_ulam, edgecolor='white', linewidth=0.5)
            ax_right.add_patch(rect_ulam)
            
            # Нижний левый - reward_t
            color_reward = cmap(norm_reward(reward_t_matrix[i, j]))
            rect_reward = Rectangle((x, y), sub_width, sub_height,
                                  facecolor=color_reward, edgecolor='white', linewidth=0.5)
            ax_right.add_patch(rect_reward)
            
            # Нижний правый - kl2_t
            color_kl2 = cmap(norm_kl2(kl2_t_matrix[i, j]))
            rect_kl2 = Rectangle((x + sub_width, y), sub_width, sub_height,
                                facecolor=color_kl2, edgecolor='white', linewidth=0.5)
            ax_right.add_patch(rect_kl2)
            
            # Добавляем значения в подячейки
            font_size = min(10, max(8, 80 / max(n_tokens, n_layers)))
            
            # Upvote (верхний левый)
            ax_right.text(x + sub_width/2, y + sub_height + sub_height/2, 
                       f'{upvote_t_matrix[i, j]:.2f}', 
                       ha='center', va='center', fontsize=font_size, weight='bold', color='black')
            
            # Ulam (верхний правый)
            ax_right.text(x + sub_width + sub_width/2, y + sub_height + sub_height/2,
                       f'{ulam_t_matrix[i, j]:.2f}',
                       ha='center', va='center', fontsize=font_size, weight='bold', color='black')
            
            # Reward (нижний левый)
            ax_right.text(x + sub_width/2, y + sub_height/2,
                       f'{reward_t_matrix[i, j]:.2f}',
                       ha='center', va='center', fontsize=font_size, weight='bold', color='black')
            
            # KL2 (нижний правый)
            ax_right.text(x + sub_width + sub_width/2, y + sub_height/2,
                       f'{kl2_t_matrix[i, j]:.3f}',
                       ha='center', va='center', fontsize=font_size, weight='bold', color='black')
            
            # Рисуем толстую белую рамку вокруг большой ячейки
            big_cell_rect = Rectangle((x, y), cell_width, cell_height, 
                                    facecolor='none', edgecolor='white', linewidth=3)
            ax_right.add_patch(big_cell_rect)
    
    # ========== НАСТРОЙКА ОСЕЙ ==========
    
    # Левая часть
    ax_left.set_xlim(-0.1, 1.1)
    ax_left.set_ylim(-0.1, n_layers + 0.1)
    ax_left.set_xticks([0.5])
    ax_left.set_xticklabels(['Layer'])
    ax_left.set_yticks(np.arange(n_layers) + 0.5)
    ax_left.set_yticklabels([f'L{layers[n_layers-1-i]}' for i in range(n_layers)])
    ax_left.set_xlabel('Layer Values')
    ax_left.set_ylabel('Layer Index')
    ax_left.set_title('Layer Analysis\nTL:Upvote TR:Ulam BL:Reward BR:KL2')
    ax_left.tick_params(top=False, right=False, labeltop=False, labelright=False)
    
    # Правая часть
    ax_right.set_xlim(-0.1, n_tokens + 0.1)
    ax_right.set_ylim(-0.1, n_layers + 0.1)
    ax_right.set_xticks(np.arange(n_tokens) + 0.5)
    ax_right.set_xticklabels([f'{i}\n{repr(token)}' for i, token in enumerate(tokens_str)], 
                           rotation=45, ha='right')
    ax_right.set_yticks(np.arange(n_layers) + 0.5)
    ax_right.set_yticklabels([f'L{layers[n_layers-1-i]}' for i in range(n_layers)])
    ax_right.set_xlabel('Token Index')
    ax_right.set_title('Token Analysis\nTL:Upvote TR:Ulam BL:Reward BR:KL2')
    ax_right.tick_params(top=False, right=False, labeltop=False, labelright=False)
    
    # Общий заголовок
    fig.suptitle('Layer-Token Knots Analysis', fontsize=16, y=0.98)
    
    # Сохраняем хитмапу
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Layer-Token knots heatmap saved: {filename}")
    print(f"  Layer values - Upvote: {upvote_l_array.min():.4f}-{upvote_l_array.max():.4f}")
    print(f"  Layer values - Ulam: {ulam_l_array.min():.4f}-{ulam_l_array.max():.4f}")
    print(f"  Layer values - Reward: {reward_l_array.min():.4f}-{reward_l_array.max():.4f}")
    print(f"  Layer values - KL2: {kl2_l_array.min():.4f}-{kl2_l_array.max():.4f}")
    print(f"  Token values - Upvote: {upvote_t_matrix.min():.4f}-{upvote_t_matrix.max():.4f}")
    print(f"  Token values - Ulam: {ulam_t_matrix.min():.4f}-{ulam_t_matrix.max():.4f}")
    print(f"  Token values - Reward: {reward_t_matrix.min():.4f}-{reward_t_matrix.max():.4f}")
    print(f"  Token values - KL2: {kl2_t_matrix.min():.4f}-{kl2_t_matrix.max():.4f}") 


if __name__ == "__main__":
    main()




