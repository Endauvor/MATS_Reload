import json
import os
from datetime import datetime
from sae6 import analyze_trajectory_origins
from graphs1 import analyze_trajectory_graph
from hooked import TransformerLensTransparentLlm
import torch
from collections import defaultdict


def generate_edges_from_trajectories(indices_per_layers: list, tokens_str: list, n_layers: int, k_top: int, T: int = None):
    """
    Finds trajectories and creates edges.
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
    
    for layer_i in range(1, n_layers):
        
        # просто ребро в самом начале
        edge = {
            "from": [0, layer_i-1],
            "to": [0, layer_i],  # К финальной позиции
            "num_trajectories": 0,
            "best_final_position": 0,
            "all_trajectories": [{"final_position": 0, "sae_index": 0}]
        }

        graph_data["edges"].append(edge)

        for token_idx in range(n_tokens):
            trajectories = trajectory_matrix[layer_i][token_idx]
            
            vertical_connection = False
            if trajectories:  # Есть траектории из этой ячейки
                
                num_trajectories = len(trajectories)
                best_final_position = min(trajectories, key=lambda t: t[0])[0]  # Ближайшая к 0 позиция
                
                # Создаем ребро к финальной позиции
                edge = {
                    "from": [token_idx, layer_i-1],
                    "to": [last_token, layer_i],  # К финальной позиции
                    "num_trajectories": num_trajectories,
                    "best_final_position": best_final_position,
                    "all_trajectories": [{"final_position": pos, "sae_index": idx} for pos, idx in trajectories]
                }
                
                graph_data["edges"].append(edge)
                total_edges += 1
                total_trajectories += num_trajectories

                if token_idx == last_token:
                    vertical_connection = True
            
        if not vertical_connection:

            # Если нет вертикального ребра для last_token, то создаем пустое ВЕРТИКАЛЬНОЕ ребро
            
            edge = {
                "from": [last_token, layer_i-1],
                "to": [last_token, layer_i],  # ВЕРТИКАЛЬНО вниз к тому же токену
                "num_trajectories": 0,
                "best_final_position": 0,
                "all_trajectories": [{"final_position": 0, "sae_index": 0}]
            }
                
            graph_data["edges"].append(edge)
            total_edges += 1

        
    
    # Добавляем статистику
    graph_data["metadata"]["total_edges"] = total_edges
    graph_data["metadata"]["total_trajectories"] = total_trajectories
    
    return graph_data

def generate_edges_from_data(data_file_path, output_dir=None, T: int=9):
    """
    Basically finds trajectories and creates edges for each length.
    """
    
    print(f"Loading SAE data from: {data_file_path}")
    
    with open(data_file_path, 'r', encoding='utf-8') as f:
        data_with_metadata = json.load(f)
    
    metadata = data_with_metadata["metadata"]
    data = data_with_metadata["data"]
    results = data_with_metadata["results"]


    #print(f"Loaded data: {len(data)} prompt lengths")
    #print(f"Original sentence: {metadata['original_sentence']}")
    #print(f"Max length: {metadata['max_prompt_length']}")
    #print(f"k_top: {metadata['k_top']}")
    
    # Параметры
    sentence_tokens = metadata["tokens"]
    max_length = metadata["max_prompt_length"]
    k_top = metadata["k_top"]
    n_layers = metadata["n_layers"]
    layers = list(range(0, n_layers))
    top_x = 10

    
    
    #data_final_top_10 = [data[max_length][l][max_length-1][:top_x] for l in layers]
    
    data_final_top_all = [] # data_final_top_all[length][layer] = [top x final indices]
    for length in range(max_length + 1):
        if length < 2:
            data_final_top_all.append([])  # пустые для 0 и 1
        else:
            top_indices = [data[length][l][length-1][:top_x] for l in layers]
            data_final_top_all.append(top_indices)
    

    #print(data_final_top_10[0])

    # Создаем структуру для хранения всех графов
    all_graphs = {
        "metadata": {
            "original_sentence": metadata["original_sentence"],
            "extended_sentence": metadata.get("extended_sentence", metadata["original_sentence"]),
            "max_prompt_length": max_length,
            "k_top": k_top,
            "sentence_tokens": sentence_tokens,
            "filter_T": T,
            "layers": layers,
            "generated_at": datetime.now().isoformat(),
            "source_data_file": data_file_path,
            "results": results,
            "data_final_top_all": data_final_top_all,
            "newcomers": []
        },
        "graphs": {}
    }
    
    
    
    #print(f"Generating graphs for lengths 2 to {max_length}...")
    
    newcomers = [
        [[] for _ in range(n_layers + 1)]  
        for _ in range(max_length + 1)     
    ]

    for length in range(2, max_length + 1):
        #print(f"  Processing length {length}...")
        graph_data = generate_edges_from_trajectories(data[length], sentence_tokens[:length], n_layers, k_top, T)
        all_graphs["graphs"][str(length)] = graph_data
        
        for edge in graph_data["edges"]:
            token_idx, layer_i = edge["from"]
            for traj in edge["all_trajectories"]:
                sae_index = traj["sae_index"]
                newcomers[length][layer_i+1].append((sae_index, token_idx))

    all_graphs["metadata"]["newcomers"] = newcomers


    
    #print(f"    Completed length {length}: {graph_data['metadata']['total_edges']} edges, ", f"{graph_data['metadata']['total_trajectories']} trajectories")
    
    # Сохраняем графы
    graph_filename = os.path.join(output_dir, f'multi_length_trajectory_graphs_T{T}.json')
    
    with open(graph_filename, 'w', encoding='utf-8') as f:
        json.dump(all_graphs, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== GRAPH GENERATION COMPLETED ===")
    print(f"Saved trajectory graphs: {graph_filename}")
    print(f"Generated graphs for {len(all_graphs['graphs'])} different prompt lengths")
    
    # Выводим сводку
    print(f"\nSummary:")
    for length in sorted(all_graphs['graphs'].keys(), key=int, reverse=True):
        graph = all_graphs['graphs'][length]
        print(f"  Length {length}: {graph['metadata']['total_edges']} edges, "
              f"{graph['metadata']['total_trajectories']} trajectories")

    return all_graphs
    
#def get_sae_descriptions(all_graphs):

#    max_length = all_graphs["metadata"]["max_prompt_length"]
#    layers = all_graphs["metadata"]["layers"]

#    newcomers = [
#        [[] for _ in range(n_layers + 1)]  
#        for _ in range(max_length + 1)     
#    ]





def main():
    

    path = "att_exp/evolution2/sae_data.json"
    output_dir = "att_exp/evolution2"
    T = 22


    
    
    print("Starting multi-length SAE trajectory and graph generation...")

    all_graphs = generate_edges_from_data(path, output_dir, T)
    G = analyze_trajectory_graph(all_graphs)

    print(f"\nGraph generation completed!")



if __name__ == "__main__":
    main() 