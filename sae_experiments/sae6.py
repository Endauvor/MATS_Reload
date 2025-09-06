"""
Analyze SAE features evolution with various functions. 
"analyze_layer_contributions_to_prediction": takes hidden at last position, patches it to the end and gets model predictions. 
"plot_indices_evolution": plots SAE features evolution per layer.
"analyze_trajectory_origins": finds trajectories origins and creates heatmaps.
... some initial functions for graph creation.
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
import json


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
    

def plot_indices_evolution(indices_per_layer: list, tokens_str: list, layers: list, k_top: int, save_dir: str = "att_exp/evolution"):
    """
    Создает графики эволюции SAE индексов для каждого слоя.
    
    Args:
        indices_per_layer: список indices для каждого слоя [layer][time_step][indices]
        tokens_str: список токенов
        layers: список номеров слоев  
        k_top: количество топ индексов
        save_dir: директория для сохранения
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for layer_i, layer_idx in enumerate(layers):
        if layer_i >= len(indices_per_layer):
            continue
            
        indices_for_layer = indices_per_layer[layer_i]
        if not indices_for_layer:
            continue
            
        print(f"Creating evolution plot for layer {layer_idx}")
        
        # Извлекаем начальный список (indices[0])
        initial_indices = set(indices_for_layer[0])
        
        # Собираем все уникальные индексы, которые когда-либо появлялись
        all_indices = set()
        for step_indices in indices_for_layer:
            all_indices.update(step_indices)
        
        # Разделяем на исходные (голубые) и новые (зеленые)
        original_indices = initial_indices
        new_indices = all_indices - initial_indices
        
        plt.figure(figsize=(max(14, len(indices_for_layer) + 2), 10))
        
        # Создаем траектории для исходных индексов (голубые)
        for idx in original_indices:
            positions = []
            time_steps = []
            
            for t, step_indices in enumerate(indices_for_layer):
                if idx in step_indices:
                    position = step_indices.index(idx)
                    positions.append(position)
                    time_steps.append(t)
                else:
                    # Индекс исчез - записываем None для разрыва
                    positions.append(None)
                    time_steps.append(t)
            
            # Рисуем сплошные линии между существующими точками
            current_segment_x = []
            current_segment_y = []
            
            for t, pos in zip(time_steps, positions):
                if pos is not None:
                    current_segment_x.append(t)
                    current_segment_y.append(pos)
                else:
                    # Рисуем накопленный сегмент если он есть
                    if len(current_segment_x) > 1:
                        plt.plot(current_segment_x, current_segment_y, 'o-', 
                                color='lightblue', linewidth=1.5, markersize=4, alpha=0.8,
                                label='Original' if idx == list(original_indices)[0] else "")
                    elif len(current_segment_x) == 1:
                        plt.plot(current_segment_x, current_segment_y, 'o', 
                                color='lightblue', markersize=4, alpha=0.8)
                    
                    # Ищем следующую точку для пунктирной линии
                    next_pos = None
                    next_t = None
                    for future_t in range(t + 1, len(time_steps)):
                        if positions[future_t] is not None:
                            next_pos = positions[future_t]
                            next_t = future_t
                            break
                    
                    # Рисуем пунктирную линию если есть следующая точка
                    if current_segment_x and next_pos is not None:
                        plt.plot([current_segment_x[-1], next_t], [current_segment_y[-1], next_pos], 
                                '--', color='lightblue', linewidth=1, alpha=0.6)
                    
                    # Начинаем новый сегмент
                    current_segment_x = []
                    current_segment_y = []
            
            # Рисуем оставшийся сегмент
            if len(current_segment_x) > 1:
                plt.plot(current_segment_x, current_segment_y, 'o-', 
                        color='lightblue', linewidth=1.5, markersize=4, alpha=0.8,
                        label='Original' if idx == list(original_indices)[0] else "")
            elif len(current_segment_x) == 1:
                plt.plot(current_segment_x, current_segment_y, 'o', 
                        color='lightblue', markersize=4, alpha=0.8)
        
        # Создаем траектории для новых индексов (цвет зависит от финальной позиции)
        for idx in new_indices:
            positions = []
            time_steps = []
            
            for t, step_indices in enumerate(indices_for_layer):
                if idx in step_indices:
                    position = step_indices.index(idx)
                    positions.append(position)
                    time_steps.append(t)
                else:
                    positions.append(None)
                    time_steps.append(t)
            
            # Определяем финальную позицию и выбираем цвет
            final_position = None
            for i in range(len(positions) - 1, -1, -1):  # Идем с конца
                if positions[i] is not None:
                    final_position = positions[i]
                    break
            
            # Выбираем цвет: если индекс доходит до конца, используем градиент синий->красный
            # иначе зеленый
            if final_position is not None and positions[-1] is not None:
                # Индекс присутствует в последнем шаге - используем цветовую шкалу
                # Позиция 0 (топ) -> красный, позиция k_top-1 (низ) -> синий
                color_ratio = final_position / (k_top - 1)  # 0 для топа, 1 для низа
                red_component = 1 - color_ratio   # Больше красного для топа
                blue_component = color_ratio      # Больше синего для низа
                line_color = (red_component, 0, blue_component)  # RGB
                linewidth = 2.5
                markersize = 5
                alpha = 0.9
                label_prefix = 'New-Final'
            else:
                # Индекс не доходит до конца - обычный зеленый
                line_color = 'green'
                linewidth = 2.5
                markersize = 5
                alpha = 0.9
                label_prefix = 'New-Temp'
            
            # Рисуем сплошные линии между существующими точками
            current_segment_x = []
            current_segment_y = []
            
            for t, pos in zip(time_steps, positions):
                if pos is not None:
                    current_segment_x.append(t)
                    current_segment_y.append(pos)
                else:
                    # Рисуем накопленный сегмент если он есть
                    if len(current_segment_x) > 1:
                        plt.plot(current_segment_x, current_segment_y, 'o-', 
                                color=line_color, linewidth=linewidth, markersize=markersize, alpha=alpha,
                                label=label_prefix if idx == list(new_indices)[0] else "")
                    elif len(current_segment_x) == 1:
                        plt.plot(current_segment_x, current_segment_y, 'o', 
                                color=line_color, markersize=markersize, alpha=alpha)
                    
                    # Ищем следующую точку для пунктирной линии
                    next_pos = None
                    next_t = None
                    for future_t in range(t + 1, len(time_steps)):
                        if positions[future_t] is not None:
                            next_pos = positions[future_t]
                            next_t = future_t
                            break
                    
                    # Рисуем пунктирную линию если есть следующая точка
                    if current_segment_x and next_pos is not None:
                        plt.plot([current_segment_x[-1], next_t], [current_segment_y[-1], next_pos], 
                                '--', color=line_color, linewidth=2, alpha=0.7)
                    
                    # Начинаем новый сегмент
                    current_segment_x = []
                    current_segment_y = []
            
            # Рисуем оставшийся сегмент
            if len(current_segment_x) > 1:
                plt.plot(current_segment_x, current_segment_y, 'o-', 
                        color=line_color, linewidth=linewidth, markersize=markersize, alpha=alpha,
                        label=label_prefix if idx == list(new_indices)[0] else "")
            elif len(current_segment_x) == 1:
                plt.plot(current_segment_x, current_segment_y, 'o', 
                        color=line_color, markersize=markersize, alpha=alpha)
        
        # Получаем финальные индексы (последний временной шаг)
        final_indices = indices_for_layer[-1] if indices_for_layer else []
        
        # Настройка графика
        plt.gca().invert_yaxis()  # Позиция 0 сверху
        plt.ylim(k_top - 0.5, -0.5)
        plt.xlim(-0.5, len(indices_for_layer) - 0.5)
        
        # Подписи осей
        x_labels = ['Initial'] + [f'T{i}\n{repr(token)}' for i, token in enumerate(tokens_str)]
        if len(x_labels) > len(indices_for_layer):
            x_labels = x_labels[:len(indices_for_layer)]
        
        plt.xticks(range(len(indices_for_layer)), x_labels, rotation=45, ha='right')
        plt.yticks(range(k_top), [f'{i}' for i in range(k_top)])
        
        # Создаем вторую ось Y справа для отображения финальных индексов
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        # НЕ инвертируем правую ось, так как левая уже инвертирована
        
        # Создаем подписи для правой оси Y с финальными индексами
        y_labels_right = []
        for pos in range(k_top):
            if pos < len(final_indices):
                feature_idx = final_indices[pos]
                y_labels_right.append(f'[{feature_idx}]')
            else:
                y_labels_right.append('')
        
        ax2.set_yticks(range(k_top))
        ax2.set_yticklabels(y_labels_right)
        ax2.set_ylabel('Final Feature Index', rotation=270, labelpad=15)
        
        plt.xlabel('Time Step (Token Index)')
        plt.ylabel('Position in Top-K List')
        plt.title(f'SAE Indices Evolution - Layer {layer_idx}')
        plt.grid(True, alpha=0.3)
        
        # Легенда
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            # Убираем дубликаты в легенде
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Добавляем информацию о количестве индексов
        plt.text(0.02, 0.98, f'Original indices: {len(original_indices)}\nNew indices: {len(new_indices)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Сохраняем основной график
        filename = os.path.join(save_dir, f'layer_{layer_idx}_indices_evolution.png')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved evolution plot: {filename}")
        print(f"  Original indices: {len(original_indices)}, New indices: {len(new_indices)}")
        
        # ========== ДОПОЛНИТЕЛЬНЫЙ ГРАФИК 1: Только исходные индексы (голубые) ==========
        plt.figure(figsize=(max(14, len(indices_for_layer) + 2), 10))
        
        # Рисуем только исходные индексы
        for idx in original_indices:
            positions = []
            time_steps = []
            
            for t, step_indices in enumerate(indices_for_layer):
                if idx in step_indices:
                    position = step_indices.index(idx)
                    positions.append(position)
                    time_steps.append(t)
                else:
                    positions.append(None)
                    time_steps.append(t)
            
            # Рисуем сплошные линии между существующими точками
            current_segment_x = []
            current_segment_y = []
            
            for t, pos in zip(time_steps, positions):
                if pos is not None:
                    current_segment_x.append(t)
                    current_segment_y.append(pos)
                else:
                    if len(current_segment_x) > 1:
                        plt.plot(current_segment_x, current_segment_y, 'o-', 
                                color='lightblue', linewidth=2, markersize=5, alpha=0.8,
                                label='Original' if idx == list(original_indices)[0] else "")
                    elif len(current_segment_x) == 1:
                        plt.plot(current_segment_x, current_segment_y, 'o', 
                                color='lightblue', markersize=5, alpha=0.8)
                    
                    # Пунктирная линия до следующей точки
                    next_pos = None
                    next_t = None
                    for future_t in range(t + 1, len(time_steps)):
                        if positions[future_t] is not None:
                            next_pos = positions[future_t]
                            next_t = future_t
                            break
                    
                    if current_segment_x and next_pos is not None:
                        plt.plot([current_segment_x[-1], next_t], [current_segment_y[-1], next_pos], 
                                '--', color='lightblue', linewidth=1.5, alpha=0.6)
                    
                    current_segment_x = []
                    current_segment_y = []
            
            # Рисуем оставшийся сегмент
            if len(current_segment_x) > 1:
                plt.plot(current_segment_x, current_segment_y, 'o-', 
                        color='lightblue', linewidth=2, markersize=5, alpha=0.8,
                        label='Original' if idx == list(original_indices)[0] else "")
            elif len(current_segment_x) == 1:
                plt.plot(current_segment_x, current_segment_y, 'o', 
                        color='lightblue', markersize=5, alpha=0.8)
        
        # Настройка графика
        plt.gca().invert_yaxis()
        plt.ylim(k_top - 0.5, -0.5)
        plt.xlim(-0.5, len(indices_for_layer) - 0.5)
        plt.xticks(range(len(indices_for_layer)), x_labels, rotation=45, ha='right')
        plt.yticks(range(k_top), [f'{i}' for i in range(k_top)])
        
        # Правая ось с финальными индексами
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(k_top))
        ax2.set_yticklabels(y_labels_right)
        ax2.set_ylabel('Final Feature Index', rotation=270, labelpad=15)
        
        plt.xlabel('Time Step (Token Index)')
        plt.ylabel('Position in Top-K List')
        plt.title(f'SAE Original Indices Evolution - Layer {layer_idx}')
        plt.grid(True, alpha=0.3)
        
        if original_indices:
            plt.legend(loc='upper right')
        
        # Сохраняем график исходных индексов
        filename_original = os.path.join(save_dir, f'layer_{layer_idx}_original_indices.png')
        plt.tight_layout()
        plt.savefig(filename_original, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved original indices plot: {filename_original}")
        
        # ========== ДОПОЛНИТЕЛЬНЫЙ ГРАФИК 2: Только новые индексы, доходящие до конца ==========
        plt.figure(figsize=(max(14, len(indices_for_layer) + 2), 10))
        
        # Отбираем только новые индексы, которые присутствуют в последнем шаге
        final_new_indices = []
        for idx in new_indices:
            # Проверяем, присутствует ли индекс в последнем временном шаге
            if indices_for_layer and idx in indices_for_layer[-1]:
                final_new_indices.append(idx)
        
        # Рисуем только новые индексы, доходящие до конца
        for idx in final_new_indices:
            positions = []
            time_steps = []
            
            for t, step_indices in enumerate(indices_for_layer):
                if idx in step_indices:
                    position = step_indices.index(idx)
                    positions.append(position)
                    time_steps.append(t)
                else:
                    positions.append(None)
                    time_steps.append(t)
            
            # Определяем финальную позицию и цвет
            final_position = None
            for i in range(len(positions) - 1, -1, -1):
                if positions[i] is not None:
                    final_position = positions[i]
                    break
            
            # Цветовая схема (синий->красный градиент)
            color_ratio = final_position / (k_top - 1)
            red_component = 1 - color_ratio
            blue_component = color_ratio
            line_color = (red_component, 0, blue_component)
            
            # Рисуем сплошные линии между существующими точками
            current_segment_x = []
            current_segment_y = []
            
            for t, pos in zip(time_steps, positions):
                if pos is not None:
                    current_segment_x.append(t)
                    current_segment_y.append(pos)
                else:
                    if len(current_segment_x) > 1:
                        plt.plot(current_segment_x, current_segment_y, 'o-', 
                                color=line_color, linewidth=2.5, markersize=5, alpha=0.9,
                                label='New-Final' if idx == final_new_indices[0] else "")
                    elif len(current_segment_x) == 1:
                        plt.plot(current_segment_x, current_segment_y, 'o', 
                                color=line_color, markersize=5, alpha=0.9)
                    
                    # Пунктирная линия до следующей точки
                    next_pos = None
                    next_t = None
                    for future_t in range(t + 1, len(time_steps)):
                        if positions[future_t] is not None:
                            next_pos = positions[future_t]
                            next_t = future_t
                            break
                    
                    if current_segment_x and next_pos is not None:
                        plt.plot([current_segment_x[-1], next_t], [current_segment_y[-1], next_pos], 
                                '--', color=line_color, linewidth=2, alpha=0.7)
                    
                    current_segment_x = []
                    current_segment_y = []
            
            # Рисуем оставшийся сегмент
            if len(current_segment_x) > 1:
                plt.plot(current_segment_x, current_segment_y, 'o-', 
                        color=line_color, linewidth=2.5, markersize=5, alpha=0.9,
                        label='New-Final' if idx == final_new_indices[0] else "")
            elif len(current_segment_x) == 1:
                plt.plot(current_segment_x, current_segment_y, 'o', 
                        color=line_color, markersize=5, alpha=0.9)
        
        # Настройка графика
        plt.gca().invert_yaxis()
        plt.ylim(k_top - 0.5, -0.5)
        plt.xlim(-0.5, len(indices_for_layer) - 0.5)
        plt.xticks(range(len(indices_for_layer)), x_labels, rotation=45, ha='right')
        plt.yticks(range(k_top), [f'{i}' for i in range(k_top)])
        
        # Правая ось с финальными индексами
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(k_top))
        ax2.set_yticklabels(y_labels_right)
        ax2.set_ylabel('Final Feature Index', rotation=270, labelpad=15)
        
        plt.xlabel('Time Step (Token Index)')
        plt.ylabel('Position in Top-K List')
        plt.title(f'SAE New Indices (Final) Evolution - Layer {layer_idx}')
        plt.grid(True, alpha=0.3)
        
        if final_new_indices:
            plt.legend(loc='upper right')
        
        # Сохраняем график новых финальных индексов
        filename_new_final = os.path.join(save_dir, f'layer_{layer_idx}_new_final_indices.png')
        plt.tight_layout()
        plt.savefig(filename_new_final, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved new final indices plot: {filename_new_final}")
        print(f"  Final new indices count: {len(final_new_indices)}")


def analyze_token_prediction_with_patching(model, tokens, layer_idx: int, last_residual_before_attn):
    """
    Патчит активацию последней позиции перед финальным unembedding
    и анализирует предсказание следующего токена
    
    Args:
        model: TransformerLensTransparentLlm модель
        tokens: токены для анализа
        layer_idx: номер слоя (не используется, для совместимости)
        last_residual_before_attn: активация для патчинга [d_model]
        
    Returns:
        dict: результаты анализа с логитами и предсказанными токенами
    """
    def save_activation(store, name):
        def hook(tensor, hook):
            store[name] = tensor.detach().clone()
        return hook

    def patch_final_resid(tensor, hook):
        # tensor: [batch, pos, d_model]
        out = tensor.clone()
        # Заменяем активацию последней позиции на нашу
        out[0, -1, :] = last_residual_before_attn
        return out

    store = {}
    last_layer = model._model.cfg.n_layers - 1  # 11 для GPT-2 small

    # ---- чистый прогон без патчинга ----
    logits_clean = model._model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (f"blocks.{last_layer}.hook_resid_post", save_activation(store, "final_resid_clean")),
        ],
    )

    # ---- прогон с патчингом ----
    logits_patched = model._model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (f"blocks.{last_layer}.hook_resid_post", patch_final_resid),
        ],
    )

    # Анализируем предсказания для последней позиции
    clean_logits_last = logits_clean[0, -1, :]  # [vocab_size]
    patched_logits_last = logits_patched[0, -1, :]  # [vocab_size]
    
    # Получаем топ-10 предсказаний для обеих версий
    clean_probs = torch.softmax(clean_logits_last, dim=-1)
    patched_probs = torch.softmax(patched_logits_last, dim=-1)
    
    clean_top_probs, clean_top_indices = torch.topk(clean_probs, k=10)
    patched_top_probs, patched_top_indices = torch.topk(patched_probs, k=10)
    
    # Конвертируем токены в строки
    clean_top_tokens = [model._model.tokenizer.decode(idx.item()) for idx in clean_top_indices]
    patched_top_tokens = [model._model.tokenizer.decode(idx.item()) for idx in patched_top_indices]
    
    # Наиболее вероятные токены
    clean_best_token = clean_top_tokens[0]
    patched_best_token = patched_top_tokens[0]
    
    # Вычисляем KL дивергенцию между распределениями
    kl_divergence = kl_from_logits(patched_logits_last, clean_logits_last).item()
    
    results = {
        'clean_logits': clean_logits_last,
        'patched_logits': patched_logits_last,
        'clean_probs': clean_probs,
        'patched_probs': patched_probs,
        'clean_top_tokens': clean_top_tokens,
        'patched_top_tokens': patched_top_tokens,
        'clean_top_probs': clean_top_probs,
        'patched_top_probs': patched_top_probs,
        'clean_best_token': clean_best_token,
        'patched_best_token': patched_best_token,
        'kl_divergence': kl_divergence,
        'prediction_changed': clean_best_token != patched_best_token
    }
    
    return results


def analyze_layer_contributions_to_prediction(model, tokens, n_layers: int = 12, log_file=None):
    """
    Анализирует вклад каждого слоя в финальное предсказание токена.
    Для каждого layer_idx берет resid_post (перед attention) и патчит в самый конец.
    
    Args:
        model: TransformerLensTransparentLlm модель
        tokens: токены для анализа
        n_layers: количество слоев для анализа
        log_file: файл для записи результатов (опционально)
        
    Returns:
        dict: результаты анализа для каждого слоя
    """
    def save_activation(store, name):
        def hook(tensor, hook):
            store[name] = tensor.detach().clone()
        return hook
    
    def log_print(message, file=None):
        """Печатает сообщение в консоль и в файл (если передан)"""
        print(message)
        if file is not None:
            file.write(message + "\n")
            file.flush()

    def patch_final_with_layer_resid(layer_activation):
        def patch_final_resid(tensor, hook):
            # tensor: [batch, pos, d_model]
            out = tensor.clone()
            # Заменяем активацию последней позиции на активацию из layer_idx
            out[0, -1, :] = layer_activation
            return out
        return patch_final_resid

    log_print("\n=== Analyzing Layer Contributions to Final Prediction ===", log_file)
    
    # Собираем resid_post для всех слоев за один проход
    store = {}
    hooks = []
    for layer_idx in range(n_layers):
        hooks.append((f"blocks.{layer_idx}.hook_resid_post", save_activation(store, f"layer_{layer_idx}_resid_post")))
    
    # Прогон только для сбора активаций (логиты не нужны)
    _ = model._model.run_with_hooks(tokens, fwd_hooks=hooks)
    
    log_print("Collected activations from all layers. Starting patching analysis...", log_file)
    
    # Теперь для каждого слоя патчим его resid_post в финал
    results = {}
    last_layer = model._model.cfg.n_layers - 1
    
    for layer_idx in range(n_layers):
        log_print(f"\nTesting layer {layer_idx} contribution...", log_file)
        
        # Берем активацию последней позиции из этого слоя
        layer_resid_post = store[f"layer_{layer_idx}_resid_post"][0, -1, :]  # [d_model]
        
        # Патчим эту активацию в финальную позицию
        patched_logits = model._model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (f"blocks.{last_layer}.hook_resid_post", patch_final_with_layer_resid(layer_resid_post)),
            ],
        )
        
        # Анализируем предсказания
        patched_logits_last = patched_logits[0, -1, :]
        patched_probs = torch.softmax(patched_logits_last, dim=-1)
        patched_top_probs, patched_top_indices = torch.topk(patched_probs, k=5)
        patched_top_tokens = [model._model.tokenizer.decode(idx.item()) for idx in patched_top_indices]
        
        results[layer_idx] = {
            'top_tokens': patched_top_tokens,
            'top_probs': patched_top_probs.tolist(),
            'best_token': patched_top_tokens[0]
        }
        
        log_print(f"  Layer {layer_idx} -> {repr(patched_top_tokens[0])} (p={patched_top_probs[0]:.4f})", log_file)
        log_print(f"  Top-3: {[repr(t) for t in patched_top_tokens[:3]]}", log_file)
        
        # Записываем детальную информацию в файл
        if log_file:
            log_print(f"  Top-5 probabilities: {[f'{p:.4f}' for p in patched_top_probs.tolist()]}", log_file)
            log_print(f"  All top-5: {patched_top_tokens}", log_file)
    
    # Печатаем сводку
    log_print(f"\n=== Summary of Layer Contributions ===", log_file)
    
    unique_predictions = {}
    for layer_idx in range(n_layers):
        token = results[layer_idx]['best_token']
        if token not in unique_predictions:
            unique_predictions[token] = []
        unique_predictions[token].append(layer_idx)
    
    log_print(f"Unique predictions found: {len(unique_predictions)}", log_file)
    for token, layers in unique_predictions.items():
        log_print(f"  {repr(token)}: layers {layers}", log_file)
    
    # Дополнительная детальная информация в файл
    if log_file:
        log_print(f"\n=== Detailed Results Table ===", log_file)
        log_print(f"{'Layer':<6} | {'Best Token':<15} | {'Probability':<12} | {'Top-3 Tokens'}", log_file)
        log_print(f"{'-'*6}+{'-'*16}+{'-'*13}+{'-'*30}", log_file)
        for layer_idx in range(n_layers):
            best_token = results[layer_idx]['best_token']
            best_prob = results[layer_idx]['top_probs'][0]
            top3 = results[layer_idx]['top_tokens'][:3]
            log_print(f"{layer_idx:<6} | {repr(best_token):<15} | {best_prob:<12.4f} | {top3}", log_file)
    
    return results


def main():
   
    MODEL_NAME = "gpt2-small"
    SENTENCE = "The final correct answer: Max, Mark and Smith are in the empty dark room. Smith left. Mark gave flashlight to" #. Mark gave flashlight to
    #SENTENCE = "It is characterized by its soulful and emotive sound, often featuring lyrics that express feelings of sadness, loss, and longing. The blues genre has evolved over the years, incorporating various styles and influences, but its core elements remain the same. From Robert Johnson to B.B. King, legendary blues musicians have made significant contributions to the genre. The blues has also had a profound impact on other genres, such as rock and roll, jazz, and rhythm and blues. Today, the blues remains a popular and enduring genre of music, with many artists continuing to create and perform blues music. The blues is a universal language, expressing emotions and experiences that transcend borders and cultures. The final correct answer: Max, Mark and Smith are in the empty dark room. Smith left. Max gave flashlight to"
    
    k_top = 40

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

    # Для каждого слоя будем собирать свой список индексов
    indices_per_layer = []

    

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
        
        # Начинаем новый список индексов для этого слоя
        layer_indices = []
        layer_indices.append(top_indices_without_attn) 

        top_indices_after_attn = top_indices_after_attn.tolist()
        
        #print("without attn", top_indices_without_attn)
        #print("with attn", top_indices_after_attn)

        upvote_l, ulam_result_l, new_reward_l = analysis_results(top_indices_without_attn, top_indices_after_attn)
        KL2_l = (kl_from_logits(acts_pre_whole, acts_pre_initial) + kl_from_logits(acts_pre_initial, acts_pre_whole)) / 2
        
        # Сохраняем значения для слоя
        upvote_l_values.append(upvote_l)
        ulam_l_values.append(1 - ulam_result_l)  # Инвертируем как в sae4.py
        reward_l_values.append(new_reward_l)
        kl2_l_values.append(KL2_l.item() if hasattr(KL2_l, 'item') else KL2_l)
        
        
        for token_idx in range(n_tokens):
            #print(f"  Processing token {token_idx}: {repr(tokens_str[token_idx])}")
            
            initial_plus_k_token_adds = last_residual_before_attn + token_adds_to_the_last[:token_idx, :].sum(dim = 0)

            _, sae_cache_initial_plus_k_token_adds = sae.run_with_cache(initial_plus_k_token_adds)
            acts_pre_initial_plus_k_token_adds = sae_cache_initial_plus_k_token_adds["hook_sae_acts_pre"]
            acts_post_initial_plus_k_token_adds = sae_cache_initial_plus_k_token_adds["hook_sae_acts_post"]

            _, top_indices_initial_plus_k_token_adds = torch.topk(acts_post_initial_plus_k_token_adds, k=k_top)
            top_indices_initial_plus_k_token_adds = top_indices_initial_plus_k_token_adds.tolist()
            layer_indices.append(top_indices_initial_plus_k_token_adds)
        print(f"-------------{layer_idx}-------------")
        print(top_indices_initial_plus_k_token_adds[:15])
        # Сохраняем индексы для этого слоя
        indices_per_layer.append(layer_indices)
        print(f"Collected {len(layer_indices)} time steps for layer {layer_idx}")
    
    # Создаем графики эволюции индексов для каждого слоя
    print(f"\nCreating evolution plots for {len(indices_per_layer)} layers...")
    plot_indices_evolution(indices_per_layer, tokens_str, layers, k_top, "att_exp/evolution")
    
    # Анализируем и создаем хитмапу истоков траекторий
    print(f"\nAnalyzing trajectory origins...")
    # Создаем две хитмапы: одну для всех траекторий, другую только для топ-10
    trajectory_matrix_all = analyze_trajectory_origins(indices_per_layer, tokens_str, layers, k_top, "att_exp/evolution")
    trajectory_matrix_top10 = analyze_trajectory_origins(indices_per_layer, tokens_str, layers, k_top, "att_exp/evolution", T=9)
    
    # Генерируем JSON файлы с данными графа
    print(f"\nGenerating trajectory graph JSON files...")
    graph_data_all = generate_trajectory_graph_json(indices_per_layer, tokens_str, layers, k_top, "att_exp/evolution")
    graph_data_top10 = generate_trajectory_graph_json(indices_per_layer, tokens_str, layers, k_top, "att_exp/evolution", T=9)
    
    # Создаем график траекторий-ленточек для слоя 9
    print(f"\nCreating trajectory ribbons for layer 9...")
    
    # Анализируем вклад каждого слоя в финальное предсказание
    with open(log_filename, 'a') as log_file:
        layer_prediction_results = analyze_layer_contributions_to_prediction(
            model, tokens_tensor, n_layers=n_layers, log_file=log_file
        )




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


def analyze_trajectory_origins(indices_per_layer: list, tokens_str: list, layers: list, k_top: int, save_dir: str = "att_exp/evolution1",  L: int = None, T: int = None):
    """
    Анализирует начальные точки новых траекторий и создает хитмапу.
    
    Args:
        indices_per_layer: список indices для каждого слоя [layer][time_step][indices]
        tokens_str: список токенов
        layers: список номеров слоев  
        k_top: количество топ индексов
        save_dir: директория для сохранения
        T: максимальная финальная позиция для отображения (0 <= final_pos <= T). Если None, показывать все.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if T is not None:
        print(f"Analyzing trajectory origins (filtering: final_pos <= {T})...")
    else:
        print(f"Analyzing trajectory origins (no filtering)...")
    
    n_layers = len(layers)
    n_tokens = len(tokens_str)
    
    # Создаем матрицу для хранения информации о траекториях
    # trajectory_matrix[layer_i][token_idx] = список (final_position, index) для траекторий, начинающихся в этой точке
    trajectory_matrix = [[[] for _ in range(n_tokens)] for _ in range(n_layers)]  # Только для токенов, без Initial
    
    for layer_i, layer_idx in enumerate(layers):
        if layer_i >= len(indices_per_layer):
            continue
            
        indices_for_layer = indices_per_layer[layer_i]
        if not indices_for_layer:
            continue
            
        print(f"  Processing layer {layer_idx}...")
        
        # Определяем исходные индексы (Initial)
        initial_indices = set(indices_for_layer[0])
        
        # Собираем все уникальные индексы
        all_indices = set()
        for step_indices in indices_for_layer:
            all_indices.update(step_indices)
        
        # Новые индексы (которые не были в Initial)
        new_indices = all_indices - initial_indices
        
        # Для каждого нового индекса найдем, где он впервые появляется и где заканчивается
        for idx in new_indices:
            first_appearance = None
            final_position = None
            
            # Ищем первое появление (начиная с токенов, пропускаем Initial)
            for t, step_indices in enumerate(indices_for_layer[1:], start=1):
                if idx in step_indices:
                    first_appearance = t
                    break
            
            # Ищем финальную позицию (если индекс присутствует в последнем шаге)
            if indices_for_layer and idx in indices_for_layer[-1]:
                final_position = indices_for_layer[-1].index(idx)
            
            # Записываем информацию о траектории (только для токенов, не для Initial)
            # Применяем фильтр по T если задан
            if first_appearance is not None and final_position is not None and first_appearance > 0:
                if T is None or final_position <= T:  # Фильтр по финальной позиции
                    token_idx = first_appearance - 1  # Приводим к индексу токена (T0=0, T1=1, ...)
                    trajectory_matrix[layer_i][token_idx].append((final_position, idx))
    
    # Создаем хитмапу
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(max(12, n_tokens + 2), max(8, n_layers + 2)))
    
    # Создаем цветовую карту от синего (low rank) к красному (high rank)
    colors = ['blue', 'purple', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('trajectory', colors, N=n_bins)
    
    cell_size = 1.0
    
    for layer_i in range(n_layers):
        for token_idx in range(n_tokens):  # Только токены
            trajectories = trajectory_matrix[layer_i][token_idx]
            
            x = token_idx
            y = layer_i  # Слой 0 внизу, слой 11 вверху
            
            if not trajectories:
                # Нет траекторий - светло-зеленый квадрат
                rect = patches.Rectangle((x, y), cell_size, cell_size, 
                                       linewidth=2, edgecolor='black', facecolor='lightgreen')
                ax.add_patch(rect)
            elif len(trajectories) == 1:
                # Одна траектория - простой цвет
                final_pos, idx = trajectories[0]
                color_value = 1.0 - (final_pos / (k_top - 1))  # 1.0 для позиции 0 (красный), 0.0 для позиции k_top-1 (синий)
                color = cmap(color_value)
                
                rect = patches.Rectangle((x, y), cell_size, cell_size, 
                                       linewidth=2, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
                
                # Добавляем номер индекса
                ax.text(x + cell_size/2, y + cell_size/2, str(idx), 
                       ha='center', va='center', fontsize=8, weight='bold', color='white')
                
            else:
                # Несколько траекторий - цвет по лучшей траектории, без штриховки
                # Выбираем наиболее "красную" траекторию (с наименьшей final_position)
                best_trajectory = min(trajectories, key=lambda t: t[0])
                final_pos, best_idx = best_trajectory
                color_value = 1.0 - (final_pos / (k_top - 1))
                color = cmap(color_value)
                
                # Основной прямоугольник
                rect = patches.Rectangle((x, y), cell_size, cell_size, 
                                       linewidth=2, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
                
                # Добавляем количество траекторий с черной подложкой
                ax.text(x + cell_size/2, y + cell_size/2, f'{len(trajectories)}', 
                       ha='center', va='center', fontsize=10, weight='bold', 
                       color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
    
    # Настройка осей
    ax.set_xlim(-0.1, n_tokens + 0.1)
    ax.set_ylim(-0.1, n_layers + 0.1)
    
    # Подписи X (токены)
    x_labels = [f'T{i}\n{repr(token)}' for i, token in enumerate(tokens_str)]
    ax.set_xticks(np.arange(n_tokens) + 0.5)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Подписи Y (слои)
    ax.set_yticks(np.arange(n_layers) + 0.5)
    ax.set_yticklabels([f'Layer {layers[i]}' for i in range(n_layers)])
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    if T is not None:
        ax.set_title(f'SAE Trajectory Origins Heatmap (Final Position ≤ {T})\n(Color: Final Position | Numbers: Multiple Trajectories)')
    else:
        ax.set_title('SAE Trajectory Origins Heatmap\n(Color: Final Position | Numbers: Multiple Trajectories)')
    
    # Добавляем цветовую шкалу
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Final Position (0=Top, Red | High=Bottom, Blue)')
    
    # Настраиваем метки цветовой шкалы для позиций
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels([f'{k_top-1}', f'{int((k_top-1)*0.75)}', f'{int((k_top-1)*0.5)}', f'{int((k_top-1)*0.25)}', '0'])
    
    # Сетка
    ax.set_xticks(np.arange(n_tokens + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_layers + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Легенда убрана для чистоты визуализации
    
    # Сохраняем хитмапу
    if T is not None:
        filename = os.path.join(save_dir, f'trajectory_origins_heatmap_T{T}_L{L}.png')
    else:
        filename = os.path.join(save_dir, f'trajectory_origins_heatmap_L{L}.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trajectory origins heatmap: {filename}")
    
    # Выводим статистику
    total_trajectories = 0
    multiple_origin_cells = 0
    
    for layer_i in range(n_layers):
        layer_trajectories = 0
        for token_idx in range(n_tokens):
            trajectories = trajectory_matrix[layer_i][token_idx]
            if trajectories:
                layer_trajectories += len(trajectories)
                total_trajectories += len(trajectories)
                if len(trajectories) > 1:
                    multiple_origin_cells += 1
        print(f"  Layer {layers[layer_i]}: {layer_trajectories} new trajectories")
    
    if T is not None:
        print(f"Total new trajectories (final_pos ≤ {T}): {total_trajectories}")
    else:
        print(f"Total new trajectories: {total_trajectories}")
    print(f"Cells with multiple trajectories: {multiple_origin_cells}")
    
    return trajectory_matrix


def generate_trajectory_graph_json(indices_per_layer: list, tokens_str: list, layers: list, k_top: int, 
                                  save_dir: str = "att_exp/evolution", T: int = None):
    """
    Создает JSON файл с информацией о графе траекторий.
    
    Args:
        indices_per_layer: список indices для каждого слоя [layer][time_step][indices]
        tokens_str: список токенов
        layers: список номеров слоев  
        k_top: количество топ индексов
        save_dir: директория для сохранения
        T: максимальная финальная позиция для отображения (0 <= final_pos <= T). Если None, показывать все.
    
    Returns:
        dict: данные графа
    """
    import json
    
    os.makedirs(save_dir, exist_ok=True)
    
    if T is not None:
        print(f"Generating trajectory graph JSON (filtering: final_pos <= {T})...")
    else:
        print(f"Generating trajectory graph JSON (no filtering)...")
    
    n_layers = len(layers)
    n_tokens = len(tokens_str)
    last_token = n_tokens - 1  # Индекс последнего токена
    
    # Создаем матрицу для хранения информации о траекториях (та же логика что в хитмапе)
    trajectory_matrix = [[[] for _ in range(n_tokens)] for _ in range(n_layers)]
    
    # Собираем траектории (копируем логику из analyze_trajectory_origins)
    for layer_i, layer_idx in enumerate(layers):
        if layer_i >= len(indices_per_layer):
            continue
            
        indices_for_layer = indices_per_layer[layer_i]
        if not indices_for_layer:
            continue
            
        print(f"  Processing layer {layer_idx} for graph...")
        
        # Определяем исходные индексы (Initial)
        initial_indices = set(indices_for_layer[0])
        
        # Собираем все уникальные индексы
        all_indices = set()
        for step_indices in indices_for_layer:
            all_indices.update(step_indices)
        
        # Новые индексы (которые не были в Initial)
        new_indices = all_indices - initial_indices
        
        # Для каждого нового индекса найдем, где он впервые появляется и где заканчивается
        for idx in new_indices:
            first_appearance = None
            final_position = None
            
            # Ищем первое появление (начиная с токенов, пропускаем Initial)
            for t, step_indices in enumerate(indices_for_layer[1:], start=1):
                if idx in step_indices:
                    first_appearance = t
                    break
            
            # Ищем финальную позицию (если индекс присутствует в последнем шаге)
            if indices_for_layer and idx in indices_for_layer[-1]:
                final_position = indices_for_layer[-1].index(idx)
            
            # Записываем информацию о траектории (применяем фильтр по T)
            if first_appearance is not None and final_position is not None and first_appearance > 0:
                if T is None or final_position <= T:  # Фильтр по финальной позиции
                    token_idx = first_appearance - 1  # Приводим к индексу токена (T0=0, T1=1, ...)
                    trajectory_matrix[layer_i][token_idx].append((final_position, idx))
    
    # Создаем структуру данных для JSON
    graph_data = {
        "metadata": {
            "n_tokens": n_tokens,
            "n_layers": n_layers,
            "k_top": k_top,
            "filter_T": T,
            "prompt_length": len(tokens_str),
            "tokens": tokens_str,
            "layers": layers
        },
        "edges": []
    }
    
    # Генерируем ребра графа
    total_edges = 0
    total_trajectories = 0
    
    for layer_i in range(n_layers):
        layer_idx = layers[layer_i]
        
        for token_idx in range(n_tokens):
            trajectories = trajectory_matrix[layer_i][token_idx]
            
            if trajectories:  # Есть траектории из этой ячейки
                # Подсчитываем статистику
                num_trajectories = len(trajectories)
                best_final_position = min(trajectories, key=lambda t: t[0])[0]  # Ближайшая к 0 позиция
                
                # Создаем ребро
                edge = {
                    "from": [token_idx, layer_idx],
                    "to": [last_token, layer_idx + 1],  # Условная следующая вершина
                    "num_trajectories": num_trajectories,
                    "best_final_position": best_final_position,
                    "all_trajectories": [{"final_position": pos, "sae_index": idx} for pos, idx in trajectories]
                }
                
                graph_data["edges"].append(edge)
                total_edges += 1
                total_trajectories += num_trajectories
    
    # Добавляем статистику в метаданные
    graph_data["metadata"]["total_edges"] = total_edges
    graph_data["metadata"]["total_trajectories"] = total_trajectories
    
    # Сохраняем JSON
    if T is not None:
        filename = os.path.join(save_dir, f'trajectory_graph_{len(tokens_str)}tokens_T{T}.json')
    else:
        filename = os.path.join(save_dir, f'trajectory_graph_{len(tokens_str)}tokens.json')
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved trajectory graph JSON: {filename}")
    if T is not None:
        print(f"  Total edges (final_pos ≤ {T}): {total_edges}")
        print(f"  Total trajectories (final_pos ≤ {T}): {total_trajectories}")
    else:
        print(f"  Total edges: {total_edges}")
        print(f"  Total trajectories: {total_trajectories}")
    
    return graph_data


if __name__ == "__main__":
    main()




