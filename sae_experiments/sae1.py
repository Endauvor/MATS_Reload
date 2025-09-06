"""
Token adds analysis with various techniqes. 
Mostly trying to answer whether by deleting of inserting an add it's possible to get something usefull.
"""

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
                if not np.isnan(value):  
                    text_color = 'white' if value > np.nanmax(head_data) * 0.5 else 'black'
                    plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                            color=text_color, fontsize=8, weight='bold')
        
        plt.xticks(range(len(tokens_str)), [f'{i}\n{repr(token)}' for i, token in enumerate(tokens_str)], 
                   rotation=45, ha='right')
        plt.yticks(range(n_layers), [f'Layer {i}' for i in range(n_layers)])
        
        plt.xlabel('Token Index')
        plt.ylabel('Layer Index')
        plt.title(f'KL Divergence Heatmap for Attention Head {head_idx + 1}')
        
        cbar = plt.colorbar(im)
        cbar.set_label('KL Divergence')
        
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


    log_filename = f"att_exp/token_ads/logs.txt"
   
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
    
    # Combine original sentence with generated tokens
    full_sentence = SENTENCE + "".join(generated_tokens)
    print(f"Full sentence: {full_sentence}")

    tokens_tensor = model._last_run.tokens[0] # Берем первое предложение из батча
    tokens_str = model.tokens_to_strings(tokens_tensor)
    
    print("\n Tokens: ")
    for i, token in enumerate(tokens_str):
        print(f"{i}: {repr(token)}")

    d_model = model._model.cfg.d_model

    # Open file for writing results
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(f"KL Divergence Analysis Results\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Sentence: {SENTENCE}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Tokens:\n")
        for i, token in enumerate(tokens_str):
            f.write(f"{i}: {repr(token)}\n")
        f.write("\n")

        # Инициализация матрицы для сбора KL дивергенций по головам и токенам для каждого слоя
        n_heads = 12
        n_layers = 11
        n_tokens = len(tokens_str)
        
        # Для каждого слоя будем хранить матрицу [head, token] с KL дивергенциями
        layer_head_token_matrices = []
        
        layer_token_kl_matrix = torch.full((n_layers, n_tokens), 0.0, dtype=torch.float32)  # [layer, token]
        
        kl_initial_whole_per_layer = []

        q = 1
        for layer_idx in range(q, n_layers):  # Цикл по первым 10 слоям (0-9)
            f.write(f"Layer {layer_idx}:\n")

            all_adds_to_the_last = model.decomposed_attn(layer=layer_idx)[-1, :, :, :]
    
            print(f"Layer {layer_idx}: adds_to_the_last.shape = {all_adds_to_the_last.shape}")
            
            #head_adds_to_the_last = all_adds_to_the_last.sum(dim = 0)

            token_adds_to_the_last = all_adds_to_the_last.sum(dim = 1)
            
            all_adds_to_the_last = all_adds_to_the_last.reshape(-1, d_model) # [n_source * n_head, d_model]

            n_adds = all_adds_to_the_last.shape[0]
           
            last_residual_before_attn = model.residual_in(layer=layer_idx)[0, -1, :]
            last_residual_after_attn = model.residual_after_attn(layer=layer_idx)[0, -1, :]
            
            #whole_hidden_after_attn = last_residual_before_attn + all_adds_to_the_last.sum(dim = 0)

            whole_hidden_after_attn = last_residual_after_attn

            print("norm before attn", torch.norm(last_residual_before_attn))
            print("difference per layer", torch.norm(last_residual_after_attn - last_residual_before_attn))
            print("norm after attn", torch.norm(last_residual_after_attn))

            #print("norms after attn:", torch.norm(whole_hidden_after_attn-last_residual_after_attn))
            whole_minus_adds = whole_hidden_after_attn.unsqueeze(0).expand(n_adds, -1) - all_adds_to_the_last
            whole_minus_token_adds = whole_hidden_after_attn.unsqueeze(0).expand(n_tokens, -1) - token_adds_to_the_last

            device = "cuda"
            release = "gpt2-small-resid-mid-v5-32k"                              
            sae_id = f"blocks.{layer_idx}.hook_resid_mid"  


            sae, cfg, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
                release=release, sae_id=sae_id, device=device
            )

           # _, sae_cache_adds = sae.run_with_cache(last_plus_add)
           # acts_pre_for_adds  = sae_cache_adds["hook_sae_acts_pre"]

            _, sae_cache_initial = sae.run_with_cache(last_residual_before_attn)
            acts_pre_initial = sae_cache_initial["hook_sae_acts_pre"]
            acts_post_initial = sae_cache_initial["hook_sae_acts_post"]

            _, sae_cache_whole = sae.run_with_cache(whole_hidden_after_attn)
            acts_pre_whole = sae_cache_whole["hook_sae_acts_pre"]
            acts_post_after_attn = sae_cache_whole["hook_sae_acts_post"]
            
            _, sae_cache_whole_minus_adds = sae.run_with_cache(whole_minus_adds)
            acts_pre_minus_add = sae_cache_whole_minus_adds["hook_sae_acts_pre"]

            _, sae_cache_whole_minus_token_adds = sae.run_with_cache(whole_minus_token_adds)
            acts_pre_minus_token_add = sae_cache_whole_minus_token_adds["hook_sae_acts_pre"]
            

            kl_whole_vs_whole_minus_add = kl_from_logits(acts_pre_minus_add, acts_pre_whole)
            kl_whole_vs_whole_minus_token_add = kl_from_logits(acts_pre_minus_token_add, acts_pre_whole)
            kl_initial_whole = kl_from_logits(acts_pre_initial, acts_pre_whole)  # [n_adds]

            analyze_sae_features(acts_post_initial, 40, "last hidden before attn", layer_idx, f)
            analyze_sae_features(acts_post_after_attn, 40, "last hidden after attn", layer_idx, f)


            # Создаем матрицу KL дивергенций для текущего слоя [head, token]
            head_token_kl_matrix = torch.full((n_heads, n_tokens), float('nan'))
            
            # Заполняем матрицу KL дивергенций по головам и токенам для текущего слоя
            # kl_whole_vs_whole_minus_add имеет размер [n_adds] где n_adds = n_tokens * n_heads
            for idx, kl_value in enumerate(kl_whole_vs_whole_minus_add):
                attention_head_idx = (idx % 12)  # 0-11 
                token_idx = idx // 12  # индекс токена
                
                # Заполняем матрицу: [head, token] для текущего слоя
                if token_idx < n_tokens:  # Проверяем границы
                    head_token_kl_matrix[attention_head_idx, token_idx] = kl_value
            
            # Сохраняем матрицу для текущего слоя
            layer_head_token_matrices.append(head_token_kl_matrix)

            
            # Заполняем матрицу KL дивергенций по слоям и токенам
            # kl_whole_vs_whole_minus_token_add имеет размер [n_tokens]
            layer_token_kl_matrix[layer_idx, :] = kl_whole_vs_whole_minus_token_add
            
            # Сохраняем kl_initial_whole для текущего слоя
            kl_initial_whole_per_layer.append(kl_initial_whole.item())
            
            # Вычисляем статистики добавок токенов
            mean_cos_sim, mean_norm, min_norm, max_norm = compute_token_adds_statistics(token_adds_to_the_last)
            
            # Вычисляем нормы для каждого токена и находим топ-5 максимальных
            token_norms = torch.norm(token_adds_to_the_last, dim=-1)  # [n_tokens]
            top5_norms_values, top5_norms_indices = torch.topk(token_norms, k=min(5, len(token_norms)), largest=True)
        
            # Записываем статистики в лог
            f.write(f"  Mean cosine similarity: {mean_cos_sim:.6f}\n")
            f.write(f"  Mean norm: {mean_norm:.6f}\n")
            f.write(f"  Min norm: {min_norm:.6f}\n")
            f.write(f"  Max norm: {max_norm:.6f}\n")
            f.write(f"  Top-5 tokens with highest norms:\n")
            for i, (norm_value, token_idx) in enumerate(zip(top5_norms_values, top5_norms_indices)):
                f.write(f"    {i+1}. Token {token_idx.item()}: {repr(tokens_str[token_idx.item()])} (norm: {norm_value:.6f})\n")

           
            top5_smallest_values, top5_indices = torch.topk(kl_whole_vs_whole_minus_add, k=5, largest = True)
            
            f.write(f"Top-5 biggest KL values:\n")
            f.write(f"Without adds we have {kl_initial_whole} \n")
            for i, (value, idx) in enumerate(zip(top5_smallest_values, top5_indices)):
                # token index (k / 12)
                attention_head_idx = (idx.item() % 12) + 1  
                token_idx = idx.item() // 12  
                f.write(f"  {i+1}. KL = {value:.6f}, attention head index = {attention_head_idx}, token index = {token_idx}")
                f.write(f", token: {repr(tokens_str[token_idx])}")
                f.write("\n")
                
            

           
        # Строим хитмапы KL дивергенций для каждого слоя (головы vs токены)
        print("\nBuilding KL divergence heatmaps for each layer (heads vs tokens)...")
        plot_layer_head_token_heatmaps(layer_head_token_matrices, tokens_str, n_layers)
        
        # Строим хитмапу KL дивергенций по слоям и токенам
        print("\nBuilding layer-token KL divergence heatmap...")
        plot_layer_token_kl_heatmap(layer_token_kl_matrix, tokens_str, n_layers, kl_initial_values=kl_initial_whole_per_layer)

    print(f"Results saved to: {log_filename}")


if __name__ == "__main__":
    main() 
