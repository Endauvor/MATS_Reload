## SAE Trajectory Graph Analysis and Visualization
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os


def trace_trajectories_backwards(json_data):
    """
    Обратный трейсинг траекторий SAE-признаков.
    Идем от финальных позиций назад к истокам.
    
    Args:
        json_data: данные из multi_length_trajectory_graphs_T9.json
        
    Returns:
        tuple: (all_vertices, all_edges) для построения графа
    """
    
    print("Starting backward trajectory tracing...")
    
    # Шаг 1: Стартуем с полной длины промпта
    max_length = max(json_data["graphs"].keys(), key=int)
    print(f"Starting from max length: {max_length}")
    
    current_vertices = []
    all_edges = []  # Инициализируем список ребер
    
    # Находим все начальные вершины ребер в полном графе И добавляем сами эти ребра
    for edge in json_data["graphs"][max_length]["edges"]:
        source_vertex = tuple(edge["from"])  # (token_idx, layer_idx)
        if source_vertex[0] == 0: #HERE
            continue
        target_vertex = tuple(edge["to"])    # (last_token, layer_idx + 1)
        
        current_vertices.append(source_vertex)
        
        # Добавляем ребро из полного графа (ведет в финальную позицию)
        all_edges.append({
            "from": source_vertex,
            "to": target_vertex,
            "edge_data": edge
        })
    
    #print(f"Found {len(current_vertices)} initial vertices in full graph")
    #print(f"Added {len(all_edges)} edges from full graph to final positions")
    
    all_vertices = set(current_vertices)  # Собираем все найденные вершины
    # Добавляем финальные вершины
    for edge in json_data["graphs"][max_length]["edges"]:
        source_vertex = tuple(edge["from"])
        if source_vertex[0] == 0: #HERE
            continue
        all_vertices.add(tuple(edge["to"]))
    step = 0
    
    # Шаг 2: Основной цикл обратного трейсинга
    while current_vertices:
        step += 1
        #print(f"\nStep {step}: Processing {len(current_vertices)} vertices: {current_vertices}")
        next_vertices = []
        
        for target_vertex in current_vertices:
            token_idx, layer_idx = target_vertex
            
            # Условия остановки: token_idx = 0 ИЛИ layer_idx = 0
            if token_idx == 0:
                #print(f"  Stopping at vertex {target_vertex} (layer_idx = 0)")
                continue
            
            if layer_idx == 0:
                #print(f"  Stopping at vertex {target_vertex} (layer_idx = 0)")
                continue
                
            # Ищем ребра в графе длины (token_idx + 1)
            target_length = str(token_idx + 1)
            if target_length not in json_data["graphs"]:
                #print(f"  No graph for length {target_length}")
                continue
            
            found_edges = 0
            #print(f"    Looking for edges in graph {target_length} that lead to token {token_idx}")
            
            
            # Ищем все ребра, которые ведут в target_vertex
            layer_found_edges = 0
            for edge in json_data["graphs"][target_length]["edges"]:
                if tuple(edge["to"]) == target_vertex:
                    source_vertex = tuple(edge["from"])
                    #print(f"      Found edge: {source_vertex} -> {target_vertex} (trajectories: {edge['num_trajectories']})")
                    if source_vertex[0] != 0:
                    #print(f"      Found edge: {source_vertex} -> {target_vertex} (trajectories: {edge['num_trajectories']})")
                        next_vertices.append(source_vertex)
                        all_vertices.add(source_vertex)
                        all_edges.append({
                            "from": source_vertex,
                            "to": target_vertex,  
                            "edge_data": edge
                        })
                        layer_found_edges += 1
                        found_edges += 1
                

            
            #if found_edges > 0:
                #print(f"  Vertex {target_vertex} <- {found_edges} total edges from graph {target_length}")
            #else:
                #print(f"  Vertex {target_vertex} <- NO EDGES found in graph {target_length}")
        
        # Переходим к следующей итерации
        current_vertices = list(set(next_vertices))  
        #print(f"  Next iteration: {len(current_vertices)} vertices: {current_vertices}")
    
    print(f"\nTracing completed!")
    print(f"Total vertices found: {len(all_vertices)}")
    print(f"Total edges found: {len(all_edges)}")
    
    return all_vertices, all_edges


def build_networkx_graph(vertices, edges):
    """
    Строит NetworkX граф из найденных вершин и ребер.
    
    Args:
        vertices: набор вершин (token_idx, layer_idx)
        edges: список ребер с данными
        
    Returns:
        nx.DiGraph: направленный граф
    """
    
    print("Building NetworkX graph...")
    
    G = nx.DiGraph()
    
    for vertex in vertices:
        token_idx, layer_idx = vertex
        G.add_node(vertex, token_idx=token_idx, layer_idx=layer_idx)
    
    for edge in edges:
        source = edge["from"]
        target = edge["to"]
        edge_data = edge["edge_data"]
        
        G.add_edge(
            source, 
            target,
            num_trajectories=edge_data["num_trajectories"],
            best_final_position=edge_data["best_final_position"]
        )
    
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def visualize_graph(G, save_path="att_exp/evolution/trajectory_graph.png", token_names=None, results=None, results1=None, results2=None):
    """
    Визуализирует граф траекторий.
    
    Args:
        G: NetworkX граф
        save_path: путь для сохранения изображения
        token_names: список названий токенов для оси X
    """
    
    print("Visualizing graph...")
    
    # Создаем позиции вершин в координатах (token_idx, layer_idx)
    pos = {}
    for node in G.nodes():
        token_idx, layer_idx = node
        pos[node] = (token_idx, layer_idx)
    
    # Настраиваем размер фигуры
    max_token = max(node[0] for node in G.nodes())
    max_layer = max(node[1] for node in G.nodes())
    
    # Увеличиваем ширину фигуры, чтобы вместить текст справа
    fig, ax = plt.subplots(figsize=(max(12, max_token + 2) + 32, max(8, max_layer + 2)))
    
    # Устанавливаем темно-серый фон
    ax.set_facecolor('#f0f0f0')
    
    # Разделяем ребра на обычные и пустые
    normal_edges = []
    empty_edges = []
    normal_edge_weights = []
    normal_edge_colors = []
    
    for source, target, data in G.edges(data=True):
        if data["num_trajectories"] == 0:
            # Пустое ребро
            empty_edges.append((source, target))
        else:
            # Обычное ребро
            normal_edges.append((source, target))
            normal_edge_weights.append(data["num_trajectories"])
            # Цвет: 0 (лучшая позиция) -> красный, большие значения -> синий
            normal_edge_colors.append(data["best_final_position"])
    
    # Нормализуем толщину обычных ребер (больше траекторий = толще, но не слишком)
    if normal_edge_weights:
        min_weight = min(normal_edge_weights)
        max_weight = max(normal_edge_weights)
        if max_weight > min_weight:
            # Больше траекторий -> толще ребра (от 1 до 4)
            normal_edge_widths = [1 + 3 * (w - min_weight) / (max_weight - min_weight) for w in normal_edge_weights]
        else:
            normal_edge_widths = [2] * len(normal_edge_weights)
    else:
        normal_edge_widths = []
    
    # Создаем цветовую карту (красный -> синий, инвертированная)
    colors = ['red', 'purple', 'blue']
    cmap = LinearSegmentedColormap.from_list('trajectory', colors, N=100)
    
    # Рисуем обычные ребра с цветовой картой
    if normal_edge_colors:
        max_position = max(normal_edge_colors) if normal_edge_colors else 1
        # Инвертируем цвета: 0 (лучшая позиция) -> 0 (красный), большие -> 1 (синий)
        if max_position > 0:
            normalized_colors = [c / max_position for c in normal_edge_colors]
        else:
            # Если все позиции равны 0, используем один цвет
            normalized_colors = [0.0] * len(normal_edge_colors)
        
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=normal_edges,
            width=normal_edge_widths,
            edge_color=normalized_colors,
            edge_cmap=cmap,
            edge_vmin=0, edge_vmax=1,
            arrows=True,
            arrowsize=5,
            arrowstyle='->',
            ax=ax
        )
    
    # Рисуем пустые ребра золотым цветом
    if empty_edges:
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=empty_edges,
            width=2,  # Обычная толщина
            edge_color='gold',
            arrows=True,
            arrowsize=5,
            arrowstyle='->',
            ax=ax
        )
    
    # Рисуем вершины
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightgray',
        node_size=300,
        alpha=0.8,
        ax=ax
    )
    
    # Подписи вершин (жирным шрифтом)
    labels = {node: f'T{node[0]}\nL{node[1]}' for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    ax.set_axis_on()
    
    # Добавляем названия токенов на нулевом слое для наглядности
    if token_names:
        token_positions = sorted(set(node[0] for node in G.nodes()))
        for pos in token_positions:
            if pos < len(token_names):
                token_name = token_names[pos].replace('Ġ', '')
                # Добавляем текст на нулевом слое
                ax.text(pos, -0.3, token_name, rotation=45, ha='right', va='top', 
                       fontsize=11, color='darkblue', fontweight='bold')
    
    # Настройка осей X с названиями токенов
    if token_names:
        # Создаем тики только для тех позиций, которые есть в графе
        token_positions = sorted(set(node[0] for node in G.nodes()))
        
        # Берем соответствующие названия токенов
        token_labels = []
        for pos in token_positions:
            if pos < len(token_names):
                # Убираем префикс "Ġ" если есть, для читаемости
                token_name = token_names[pos].replace('Ġ', '')
                token_labels.append(token_name)
            else:
                token_labels.append(f'T{pos}')
        
        # Устанавливаем тики и лейблы
        ax.set_xticks(token_positions)
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Tokens', fontsize=11, fontweight='bold')
    else:
        ax.set_xlabel('Token Index', fontsize=10, fontweight='bold')
    
    # Настройка осей Y с четкой индексацией слоев
    layer_positions = sorted(set(node[1] for node in G.nodes()))
    ax.set_yticks(layer_positions)
    ax.set_yticklabels([f'Layer {l}' for l in layer_positions], fontsize=10)
    ax.set_ylabel('Layer Index', fontsize=11, fontweight='bold')
    
    # Используем ax.text для добавления нескольких аннотаций справа от графика
    
    # Длина списка токенов нужна для фильтрации
    last_token = len(token_names) - 1

    for layer in layer_positions[:-1]:
        str_layer = str(layer)
        
        # 1. results (топовые токены) - зеленым
        if results and str_layer in results and "top_tokens" in results[str_layer] and results[str_layer]["top_tokens"]:
            top_tokens = results[str_layer]["top_tokens"][:2]
            label_text = ", ".join([token.replace('Ġ', '') for token in top_tokens])
            ax.text(1.02, layer, label_text, transform=ax.get_yaxis_transform(), 
                    fontsize=9, color='darkgreen', fontweight='bold',
                    verticalalignment='center', horizontalalignment='left')

        # 2. results2 (список таплов) - красным
        if results2:
            # Фильтруем таплы, которые не нужно отображать
            filtered_tuples = [
                t for t in results2[layer] 
                if t != (0, 0) and t != (0, last_token)
            ]
            
            tuples_to_show = filtered_tuples[:5]
            
            if tuples_to_show:
                label_text = ", ".join(map(str, tuples_to_show))
                ax.text(1.18, layer, label_text, transform=ax.get_yaxis_transform(), 
                        fontsize=9, color='red', fontweight='bold',
                        verticalalignment='center', horizontalalignment='left')

        # 3. results1 (список чисел) - синим
        #if results1:
        #    numbers = results1[layer][:5]
            # Используем более компактное представление списка
        #    label_text = repr(numbers)
        #    ax.text(1.32, layer, label_text, transform=ax.get_yaxis_transform(), 
        #            fontsize=9, color='blue', fontweight='bold',
        #            verticalalignment='center', horizontalalignment='left')

    # Настройки отображения тиков
    ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1)
    ax.tick_params(axis='x', rotation=45)
    
    # Заголовок и сетка
    ax.set_title('SAE Trajectory Graph\n(Edge thickness: more trajectories = thicker)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Расширяем пределы по Y, чтобы было видно названия токенов внизу
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0] - 0.8, current_ylim[1])
    
    # Легенда для толщины
    if normal_edge_weights:
        legend_elements = []
        # Показываем примеры: минимальное и максимальное количество траекторий
        for w in [min(normal_edge_weights), max(normal_edge_weights)]:
            # Обновленная толщина
            width = 1 + 3 * (w - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 2
            legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=width, 
                                            label=f'{w} trajectories'))
        ax.legend(handles=legend_elements, loc='upper left')
    
    # Сохраняем график
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Оставляем место справа для аннотаций
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Graph visualization saved: {save_path}")


def analyze_trajectory_graph(json_data):
    """
    Основная функция анализа графа траекторий.
    
    Args:
        json_file_path: путь к JSON файлу с данными графов
    """
    
    #print(f"Loading JSON data from: {json_file_path}")
    
    # Загружаем данные
    #with open(json_file_path, 'r', encoding='utf-8') as f:
    #    json_data = json.load(f)
    
    #print(f"Loaded data for {len(json_data['graphs'])} prompt lengths")
    #print(f"Metadata: {json_data['metadata']}")
    
    # Получаем названия токенов из максимальной длины промпта
    max_length = max(json_data["graphs"].keys(), key=int)
    token_names = json_data["graphs"][max_length]["tokens"]
    print(f"Token names loaded: {len(token_names)} tokens")
    
    # Трейсинг траекторий
    vertices, edges = trace_trajectories_backwards(json_data)
    
    # Построение NetworkX графа
    G = build_networkx_graph(vertices, edges)
    
    # Анализ графа
    #print(f"\n=== Graph Analysis ===")
    #print(f"Nodes: {G.number_of_nodes()}")
    #print(f"Edges: {G.number_of_edges()}")
    #print(f"Weakly connected components: {nx.number_weakly_connected_components(G)}")
    
    # Находим истоки (вершины без входящих ребер)
    sources = [node for node in G.nodes() if G.in_degree(node) == 0]
    #print(f"Source nodes (no incoming edges): {len(sources)}")
    
    # Находим финальные вершины (без исходящих ребер)
    sinks = [node for node in G.nodes() if G.out_degree(node) == 0]
    #print(f"Sink nodes (no outgoing edges): {len(sinks)}")
    
    # Визуализация с названиями токенов
    # results находится внутри ключа metadata
    metadata = json_data.get("metadata", {})
    results = metadata.get("results")
    results1 = metadata.get("data_final_top_all")
    results2 = metadata.get("newcomers")
    
    
    visualize_graph(G, token_names=token_names, results=results, results1 = results1, results2 = results2)
    


    # Создаем plotly версию
    try:
        from plotly_graph import create_plotly_graph
        plotly_save_path = "att_exp/evolution/trajectory_graph_plotly.html"
        create_plotly_graph(G, save_path=plotly_save_path, token_names=token_names, results=results, results1=results1, results2=results2)
    except ImportError:
        print("\nSkipping plotly graph generation: plotly library not found.")
        print("Install it with: pip install plotly")
    except Exception as e:
        print(f"\nAn error occurred during plotly graph generation: {e}")

    return G


def main():
    """
    Основная функция для анализа графа траекторий.
    """
    
    json_file_path = "att_exp/evolution2/multi_length_trajectory_graphs_T10.json"
    
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found: {json_file_path}")
        print("Please run sae8.py first to generate the data.")
        return
    
    # Анализируем граф
    G = analyze_trajectory_graph(json_file_path)
    
    print("\nGraph analysis completed!")


if __name__ == "__main__":
    main() 