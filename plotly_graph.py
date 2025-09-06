import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from api_gemma import get_feature_label_gemma_65k


def create_plotly_graph(G, save_path="att_exp/evolution/trajectory_graph_plotly.html", token_names=None, results=None, results1=None, results2=None):
    """
    Создает интерактивную визуализацию графа траекторий с помощью plotly.
    
    Args:
        G: NetworkX граф
        save_path: Путь для сохранения HTML файла
        token_names: Список названий токенов для фильтрации в results2
        results, results1, results2: Данные для всплывающих подсказок
    """
    print("Creating plotly interactive graph...")
    
    # --- 1. Подготавливаем позиции узлов (как в graphs1.py) ---
    pos = {}
    for node in G.nodes():
        token_idx, layer_idx = node
        # Увеличиваем вертикальное расстояние между слоями более умно
        pos[node] = (token_idx, layer_idx * 3)  # Увеличиваем масштаб
    
    # --- 2. Подготавливаем данные для ребер ---
    # Логика расчета параметров ребер (точно как в graphs1.py)
    normal_edges = []
    empty_edges = []
    normal_edge_weights = []
    normal_edge_colors = []
    
    for source, target, data in G.edges(data=True):
        if data["num_trajectories"] == 0:
            empty_edges.append((source, target))
        else:
            normal_edges.append((source, target))
            normal_edge_weights.append(data["num_trajectories"])
            normal_edge_colors.append(data["best_final_position"])
    
    # Нормализация толщины ребер
    if normal_edge_weights:
        min_weight = min(normal_edge_weights)
        max_weight = max(normal_edge_weights)
        if max_weight > min_weight:
            normal_edge_widths = [1 + 3 * (w - min_weight) / (max_weight - min_weight) for w in normal_edge_weights]
        else:
            normal_edge_widths = [2] * len(normal_edge_weights)
    else:
        normal_edge_widths = []
    
    # Нормализация цветов ребер
    if normal_edge_colors:
        max_position = max(normal_edge_colors) if normal_edge_colors else 1
        if max_position > 0:
            normalized_colors = [c / max_position for c in normal_edge_colors]
        else:
            normalized_colors = [0.0] * len(normal_edge_colors)
    else:
        normalized_colors = []

    # --- 3. Подготавливаем данные для узлов ---
    node_x = []
    node_y = []
    node_text = []
    node_hovertext = []
    
    # Определяем максимальный индекс токена для проверки последнего токена
    max_token_idx = max(node[0] for node in G.nodes()) if G.nodes() else 0
    
    for node in G.nodes():
        token_idx, layer_idx = node
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Текст внутри узла (как в graphs1.py)
        node_text.append(f'T{token_idx}<br>L{layer_idx}')
        
        # Всплывающая подсказка
        hover_text = f"<b>Node: ({token_idx}, {layer_idx})</b><br><br>"
        str_layer = str(layer_idx)
        
        # Для последнего токена добавляем results (топовые токены) - зеленым
        if token_idx == max_token_idx and results and str_layer in results and "top_tokens" in results[str_layer] and results[str_layer]["top_tokens"]:
            top_tokens = results[str_layer]["top_tokens"][:3]
            tokens_str = "<br>".join([t.replace('Ġ', '') for t in top_tokens])  # Каждый с новой строки
            hover_text += f"<span style='color:darkgreen'><b>Top Tokens:</b><br>{tokens_str}</span><br><br>"
        
        # Для всех вершин где token_idx >= 1
        if token_idx >= 1:
            length = token_idx + 1
            
            # results2 (таплы) - красным, каждый кортеж с новой строки (ПЕРВЫМ)
            if results2 and length < len(results2) and layer_idx < len(results2[length]):
                tuples_list = results2[length][layer_idx]
                if tuples_list:
                    # Фильтруем кортежи (убираем (0,0) и (0,token_idx))
                    filtered_tuples = [t for t in tuples_list if t != (0, 0) and t != (0, token_idx) and t[1] != 0]
                    if filtered_tuples:
                        # Формируем детальную информацию для каждого кортежа
                        detailed_features = []
                        for t in filtered_tuples:
                            sae_idx, idx = t
                            try:
                                feature_info = get_feature_label_gemma_65k(layer_idx, sae_idx)
                                if feature_info != "Req_Err":
                                    positive_logits, explanation = feature_info
                                    # Форматируем информацию о признаке
                                    feature_text = f"({sae_idx}, {idx}):<br>"
                                    feature_text += f"positive logits: {positive_logits}<br>"
                                    feature_text += f"explanation: {explanation}"
                                    detailed_features.append(feature_text)
                                    print("success")
                                else:
                                    # Если ошибка, показываем только кортеж
                                    detailed_features.append(f"({sae_idx}, {idx}): [Error getting info]")
                            except Exception:
                                # Если функция недоступна или ошибка, показываем только кортеж
                                detailed_features.append(f"({sae_idx}, {idx}): [Info unavailable]")
                        
                        if detailed_features:
                            features_str = "<br><br>".join(detailed_features)  # Разделяем признаки двойным переносом
                            hover_text += f"<span style='color:red'><b>New Features:</b><br>{features_str}</span><br><br>"
            
            # results1 (числа) - синим, каждое число с новой строки (ВТОРЫМ)
            if results1 and length < len(results1) and layer_idx < len(results1[length]):
                numbers = results1[length][layer_idx][:5]
                if numbers:
                    numbers_str = "<br>".join([str(num) for num in numbers])  # Каждое число с новой строки
                    hover_text += f"<span style='color:blue'><b>Top 5 Features:</b><br>{numbers_str}</span><br><br>"
        
        node_hovertext.append(hover_text)
    
    # --- 4. Создаем граф в plotly ---
    fig = go.Figure()
    
    # Добавляем обычные ребра (каждое с индивидуальными параметрами)
    for i, (source, target) in enumerate(normal_edges):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Цвет: красный (0) -> фиолетовый -> синий (1)
        color_val = normalized_colors[i]
        if color_val <= 0.5:
            # От красного к фиолетовому
            r = 255
            g = int(color_val * 2 * 128)  # 0 -> 128
            b = int(color_val * 2 * 255)  # 0 -> 255
        else:
            # От фиолетового к синему
            r = int(255 - (color_val - 0.5) * 2 * 255)  # 255 -> 0
            g = int(128 - (color_val - 0.5) * 2 * 128)  # 128 -> 0
            b = 255
        
        edge_color = f'rgb({r}, {g}, {b})'
        width = normal_edge_widths[i]
        
        # Добавляем каждое ребро как отдельный trace
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=width, color=edge_color),
            hovertext=f"Trajectories: {normal_edge_weights[i]}, Best Position: {normal_edge_colors[i]}",
            hoverinfo='text',
            showlegend=False,
            name=f'edge_{i}'
        ))
    
    # Добавляем пустые ребра (золотые)
    for source, target in empty_edges:
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=2, color='gold'),
            hovertext="Empty edge (0 trajectories)",
            hoverinfo='text',
            showlegend=False,
            name='empty_edge'
        ))
    
    # Добавляем узлы (очень маленькие, светло-зеленые)
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',  # Убираем текст, оставляем только маркеры
        marker=dict(
            size=6,  # Еще меньше
            color='lightgreen',  # Светло-зеленый цвет
            line=dict(width=1, color='gray')  # Серые границы (не darkgray)
        ),
        hovertext=node_hovertext,
        hoverinfo='text',
        name='nodes',
        showlegend=False
    ))
    
    # Добавляем цветовую шкалу для объяснения цветов ребер
    if normal_edge_colors:
        max_position = max(normal_edge_colors) if normal_edge_colors else 1
        
        # Создаем невидимый scatter для цветовой шкалы
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # Невидимые точки
            mode='markers',
            marker=dict(
                size=0.1,
                colorscale=[[0, 'rgb(255,0,0)'], [0.5, 'rgb(255,128,255)'], [1, 'rgb(0,0,255)']],  # красный->фиолетовый->синий (как в оригинале)
                cmin=0,
                cmax=max_position,
                colorbar=dict(
                    title=dict(
                        text="Best Final Position",
                        side="right"
                    ),
                    tickmode="linear",
                    tick0=0,
                    dtick=max(1, max_position // 5),
                    x=1.08,  # Дальше от графика (было 1.02)
                    len=0.8,
                    thickness=12,  # Тоньше (было 20)
                    tickvals=list(range(0, max_position + 1, max(1, max_position // 5))),
                    ticktext=[str(max_position - i) for i in range(0, max_position + 1, max(1, max_position // 5))]  # Переворачиваем значения
                )
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # --- 5. Настройка осей и макета ---
    # Названия токенов для оси X (как в graphs1.py)
    if token_names:
        token_positions = sorted(set(node[0] for node in G.nodes()))
        token_labels = []
        for pos_val in token_positions:
            if pos_val < len(token_names):
                token_name = token_names[pos_val].replace('Ġ', '')
                # Добавляем нумерацию: "0: токен", "1: токен", и т.д.
                token_labels.append(f"{pos_val}: {token_name}")
            else:
                token_labels.append(f'T{pos_val}')
        
        fig.update_xaxes(
            tickvals=token_positions,
            ticktext=token_labels,
            title_text="Tokens"
        )
    else:
        # Если нет названий токенов, просто показываем номера
        token_positions = sorted(set(node[0] for node in G.nodes()))
        fig.update_xaxes(
            tickvals=token_positions,
            ticktext=[f"{pos}" for pos in token_positions],
            title_text="Token Index"
        )
    
    # Настройка оси Y (с правильным масштабированием)
    original_layers = sorted(set(node[1] for node in G.nodes()))
    scaled_layer_positions = [layer * 3 for layer in original_layers]  # Те же умноженные позиции
    
    # Принудительно задаем диапазон Y, чтобы не было автомасштабирования
    min_y = min(scaled_layer_positions) - 1
    max_y = max(scaled_layer_positions) + 1
    
    fig.update_yaxes(
        tickvals=scaled_layer_positions,
        ticktext=[f'Layer {l}' for l in original_layers],
        title_text="Layer Index",
        range=[min_y, max_y]  # Принудительно задаем диапазон
    )
    
    # Общие настройки макета
    fig.update_layout(
        title="SAE Trajectory Graph (Interactive)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=100,t=40),  # Увеличиваем правый отступ еще больше
        xaxis=dict(showgrid=True, zeroline=False, gridcolor='lightgray'),  # Темнее линии сетки
        yaxis=dict(showgrid=True, zeroline=False, gridcolor='lightgray'),  # Темнее линии сетки
        height=1500,  # Увеличиваем высоту
        width=1400,   # Уменьшаем ширину (было 1800)
        autosize=False,  # Отключаем автоматическое изменение размера
        hoverlabel=dict(
            bgcolor="lightgray",  # Светло-серый фон для всплывающих подсказок
            font_size=12,
            font_family="Arial"
        )
    )
    
    # --- 6. Сохраняем файл ---
    fig.write_html(save_path)
    print(f"Plotly interactive graph saved: {save_path}") 