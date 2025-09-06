## SAE Trajectory Graph Generation - Multi-Length Analysis
import torch
import numpy as np
from hooked import TransformerLensTransparentLlm
from sae_lens import SAE, HookedSAETransformer
import os
import json
from datetime import datetime


def generate_trajectory_graph_json(indices_per_layers: list, tokens_str: list, n_layers: int, k_top: int, T: int = None):
    """
    Создает JSON файл с информацией о графе траекторий (без визуализации).
    Адаптировано из sae6.py
    """
    
    n_tokens = len(tokens_str)
    last_token = n_tokens - 1  # Индекс последнего токена
    
    trajectory_matrix = [[[] for _ in range(n_tokens)] for _ in range(n_layers)]
    
    # Собираем траектории (копируем логику из analyze_trajectory_origins)
    for layer_i in range(n_layers):
        
        indices_for_layer = indices_per_layers[layer_i]
        
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
                if final_position <= T:  # Фильтр по финальной позиции
                    token_idx = first_appearance - 1  # Приводим к индексу токена (T0=0, T1=1, ...)
                    print("token_idx", token_idx)
                    print("n_tokens", n_tokens)
                    trajectory_matrix[layer_i][token_idx].append((final_position, idx))
    
    # Создаем структуру данных для JSON
    graph_data = {
        "prompt_length": n_tokens,
        "tokens": tokens_str,
        "metadata": {
            "n_tokens": n_tokens,
            "n_layers": n_layers,
            "k_top": k_top,
            "filter_T": T
        },
        "edges": []
    }
    
    # Генерируем ребра графа
    total_edges = 0
    total_trajectories = 0
    
    for layer_i in range(n_layers):
        
        for token_idx in range(n_tokens):
            trajectories = trajectory_matrix[layer_i][token_idx]
            
            if trajectories:  # Есть траектории из этой ячейки
                # Подсчитываем статистику
                num_trajectories = len(trajectories)
                best_final_position = min(trajectories, key=lambda t: t[0])[0]  # Ближайшая к 0 позиция
                
                # Создаем ребро
                edge = {
                    "from": [token_idx, layer_i],
                    "to": [last_token, layer_i + 1],  # Условная следующая вершина
                    "num_trajectories": num_trajectories,
                    "best_final_position": best_final_position,
                    "all_trajectories": [{"final_position": pos, "sae_index": idx} for pos, idx in trajectories]
                }
                
                graph_data["edges"].append(edge)
                total_edges += 1
                total_trajectories += num_trajectories
    
    # Добавляем статистику
    graph_data["metadata"]["total_edges"] = total_edges
    graph_data["metadata"]["total_trajectories"] = total_trajectories
    
    return graph_data


def load_sae(layer_idx):


    release = "gpt2-small-resid-mid-v5-32k"                              
    sae_id = f"blocks.{layer_idx}.hook_resid_mid"  

    sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release, sae_id=sae_id, device="cuda"
    )

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
    


def analyze_prompt_length(model, all_graphs, sentence_tokens: list, layers: int, k_top: int, T: int = None):
   
    max_length = len(sentence_tokens)
    data = [[[] for _ in range(layers)] for _ in range(max_length + 1)]
    
        
    for layer_idx in range(layers):

        print(f"      Processing layer {layer_idx}...")
        residual_in = model.residual_in(layer=layer_idx)
        residual_after_attn = model.residual_after_attn(layer=layer_idx)

        sae = load_sae(layer_idx)

        for query_pos in range(1, max_length):

            decomposed_attention_q = model.decomposed_attn_q(layer=layer_idx, query_pos = query_pos)
            #print("decomp_shape", decomposed_attention_q.shape)
            # ------- without adds
            last_residual_before_attn = residual_in[0, query_pos, :]
            last_residual_after_attn = residual_after_attn[0, query_pos, :]

            

            top_indices_without_attn = run_sae(sae, last_residual_before_attn, k_top)

            # ------- with adds
            token_adds_to_the_last = decomposed_attention_q.sum(dim=1)
            
            сumulative_adds = torch.cumsum(token_adds_to_the_last, dim=0)

            #print(torch.norm(last_residual_after_attn - (last_residual_before_attn + сumulative_adds[-1,:])))

            top_indices = run_sae(sae, last_residual_before_attn + сumulative_adds, k_top)
            top_indices.insert(0, top_indices_without_attn)

            length = query_pos + 1
            data[query_pos + 1][layer_idx] = top_indices
                

    for length in range(2, max_length + 1):
        
        graph_data = generate_trajectory_graph_json(data[length], sentence_tokens[:length], layers, k_top, T)
        all_graphs["graphs"][str(length)] = graph_data
        print(f"  Completed length {length}: {graph_data['metadata']['total_edges']} edges, {graph_data['metadata']['total_trajectories']} trajectories")
    
    
    return all_graphs  


def main():
    MODEL_NAME = "gpt2-small"
    SENTENCE = "The final correct answer: Max, Mark and Smith are in the empty dark room. Smith left. Mark gave flashlight to"
    k_top = 40
    T = 10 # Фильтр для топ-10

    print("Starting multi-length SAE trajectory graph generation...")
    
    # Инициализируем модель
    model = TransformerLensTransparentLlm(
        model_name=MODEL_NAME, 
        device="cuda",
        dtype=torch.float32
    )

    # Запускаем на полном предложении чтобы получить токены
    model.run([SENTENCE])
    full_tokens_str = model.tokens_to_strings(model._last_run.tokens[0])
    
    print(f"\nFull sentence tokens ({len(full_tokens_str)}):")
    for i, token in enumerate(full_tokens_str):
        print(f"{i}: {repr(token)}")

    max_length = len(full_tokens_str)
    n_layers = 12
    layers = list(range(0, n_layers))
    
    # Создаем структуру для хранения всех графов
    all_graphs = {
        "metadata": {
            "original_sentence": SENTENCE,
            "max_prompt_length": max_length,
            "k_top": k_top,
            "filter_T": T,
            "layers": layers,
            "generated_at": datetime.now().isoformat()
        },
        "graphs": {}
    }
    

    save_dir = "att_exp/evolution"
    os.makedirs(save_dir, exist_ok=True)
    all_graphs = analyze_prompt_length(model, all_graphs, full_tokens_str, n_layers, k_top, T)

    filename = os.path.join(save_dir, f'multi_length_trajectory_graphs_T{T}.json')
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_graphs, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== COMPLETED ===")
    print(f"Saved multi-length trajectory graphs: {filename}")
    print(f"Generated graphs for {len(all_graphs['graphs'])} different prompt lengths")
    
    # Выводим сводку
    print(f"\nSummary:")
    for length in sorted(all_graphs['graphs'].keys(), key=int, reverse=True):
        graph = all_graphs['graphs'][length]
        print(f"  Length {length}: {graph['metadata']['total_edges']} edges, {graph['metadata']['total_trajectories']} trajectories")


if __name__ == "__main__":
    main()