"""
Script để vẽ subgraph thu nhỏ từ đồ thị tổng quát
Hiển thị một phần đại diện của đồ thị để dễ quan sát cấu trúc
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os

def load_graph(edge_list_path='data/Lab_graph_edge_list.csv'):
    """Load đồ thị từ file edge list"""
    try:
        G = nx.read_edgelist(
            edge_list_path,
            data=(('weight', float), ('relation', str))
        )
        print(f"✓ Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except FileNotFoundError:
        print(f"❌ File not found: {edge_list_path}")
        print("👉 Run Lab_Build_graph.py first")
        return None

def extract_sample_subgraph(G, num_patients=3, seed=42):
    """
    Trích xuất subgraph mẫu với một số bệnh nhân
    """
    np.random.seed(seed)
    
    # Lấy tất cả các node theo type
    patient_nodes = [n for n in G.nodes() if G.nodes.get(n, {}).get('type') == 'Patient']
    lab_nodes = [n for n in G.nodes() if G.nodes.get(n, {}).get('type') == 'Lab_Index']
    disease_nodes = [n for n in G.nodes() if G.nodes.get(n, {}).get('type') == 'Disease']
    
    if len(patient_nodes) == 0:
        # Nếu không có attribute 'type', infer từ cấu trúc
        patient_nodes = [n for n in G.nodes() if len(n) > 10 and '.' in n]  # Patient IDs
        lab_nodes = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'WBC', 
                     'NEUT', 'EO', 'BASO', 'MONO', 'LYMPH', 'MPV', 'PCT', 'PDW']
        disease_nodes = [n for n in G.nodes() if n not in patient_nodes and n not in lab_nodes]
    
    print(f"  Total patients: {len(patient_nodes)}")
    print(f"  Total lab indices: {len(lab_nodes)}")
    print(f"  Total diseases: {len(disease_nodes)}")
    
    # Chọn ngẫu nhiên một số bệnh nhân
    selected_patients = np.random.choice(patient_nodes, 
                                        size=min(num_patients, len(patient_nodes)), 
                                        replace=False)
    
    # Lấy tất cả neighbors của các bệnh nhân đã chọn
    subgraph_nodes = set(selected_patients)
    for patient in selected_patients:
        neighbors = list(G.neighbors(patient))
        subgraph_nodes.update(neighbors)
    
    # Tạo subgraph
    subgraph = G.subgraph(subgraph_nodes).copy()
    
    print(f"\n✓ Extracted subgraph:")
    print(f"  Patients: {len([n for n in subgraph.nodes() if n in patient_nodes])}")
    print(f"  Lab indices: {len([n for n in subgraph.nodes() if n in lab_nodes])}")
    print(f"  Diseases: {len([n for n in subgraph.nodes() if n in disease_nodes])}")
    print(f"  Total nodes: {subgraph.number_of_nodes()}")
    print(f"  Total edges: {subgraph.number_of_edges()}")
    
    return subgraph

def visualize_subgraph(subgraph, output_path='output/subgraph_visualization.png'):
    """
    Vẽ subgraph với layout đẹp và màu sắc phân biệt
    """
    os.makedirs('output', exist_ok=True)
    
    # Phân loại nodes
    lab_nodes = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'WBC', 
                 'NEUT', 'EO', 'BASO', 'MONO', 'LYMPH', 'MPV', 'PCT', 'PDW']
    
    patient_nodes = []
    lab_nodes_present = []
    disease_nodes = []
    
    for node in subgraph.nodes():
        if node in lab_nodes:
            lab_nodes_present.append(node)
        elif len(str(node)) > 10 and '.' in str(node):
            patient_nodes.append(node)
        else:
            disease_nodes.append(node)
    
    # Tạo layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Vẽ edges với màu và độ dày theo relation
    edge_colors = []
    edge_widths = []
    
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', '')
        weight = data.get('weight', 1)
        
        if relation == 'Lab_Value':
            edge_colors.append('#3498db')  # Blue
            edge_widths.append(0.5)
        elif relation == 'Have_Disease':
            if weight == 1:
                edge_colors.append('#e74c3c')  # Red (primary)
                edge_widths.append(2.5)
            else:
                edge_colors.append('#f39c12')  # Orange (secondary)
                edge_widths.append(1.5)
        else:
            edge_colors.append('gray')
            edge_widths.append(0.5)
    
    nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, 
                           width=edge_widths, alpha=0.6, ax=ax)
    
    # Vẽ nodes với màu phân biệt
    # 1. Patient nodes (green)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=patient_nodes,
                           node_color='#2ecc71', node_size=800,
                           node_shape='o', alpha=0.9, ax=ax,
                           edgecolors='black', linewidths=2)
    
    # 2. Lab nodes (blue)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=lab_nodes_present,
                           node_color='#3498db', node_size=600,
                           node_shape='s', alpha=0.9, ax=ax,
                           edgecolors='black', linewidths=1.5)
    
    # 3. Disease nodes (red)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=disease_nodes,
                           node_color='#e74c3c', node_size=700,
                           node_shape='^', alpha=0.9, ax=ax,
                           edgecolors='black', linewidths=2)
    
    # Vẽ labels
    labels = {}
    for node in subgraph.nodes():
        if node in patient_nodes:
            # Rút gọn Patient ID
            labels[node] = f"P-{str(node)[-4:]}"
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=9,
                           font_weight='bold', font_color='white', ax=ax)
    
    # Thêm legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
                   markersize=15, label='Patient', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db',
                   markersize=15, label='Lab Index', markeredgecolor='black', markeredgewidth=1.5),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#e74c3c',
                   markersize=15, label='Disease (ICD)', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], color='#3498db', linewidth=2, label='Lab Value'),
        plt.Line2D([0], [0], color='#e74c3c', linewidth=3, label='Primary Disease (w=1)'),
        plt.Line2D([0], [0], color='#f39c12', linewidth=2, label='Secondary Disease (w=0.5)'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
             framealpha=0.9, edgecolor='black')
    
    # Title và styling
    ax.set_title('Medical Knowledge Graph - Sample Subgraph\n(Heterogeneous Graph Structure)',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add statistics box
    stats_text = (f"Nodes: {subgraph.number_of_nodes()} "
                 f"(P:{len(patient_nodes)}, L:{len(lab_nodes_present)}, D:{len(disease_nodes)})\n"
                 f"Edges: {subgraph.number_of_edges()}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    plt.show()
    plt.close()

def create_schematic_diagram(output_path='output/graph_schema.png'):
    """
    Tạo biểu đồ minh họa cấu trúc tổng quát của đồ thị
    """
    os.makedirs('output', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define positions
    positions = {
        'patient': (0.5, 0.5),
        'lab1': (0.2, 0.8),
        'lab2': (0.3, 0.9),
        'lab3': (0.4, 0.85),
        'disease1': (0.7, 0.7),
        'disease2': (0.8, 0.6)
    }
    
    # Draw nodes
    patient_circle = plt.Circle(positions['patient'], 0.08, color='#2ecc71', 
                                ec='black', linewidth=2, zorder=3)
    ax.add_patch(patient_circle)
    ax.text(positions['patient'][0], positions['patient'][1], 'Patient\n(Node)', 
           ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Lab nodes
    for i, (key, pos) in enumerate([(k, v) for k, v in positions.items() if 'lab' in k]):
        rect = Rectangle((pos[0]-0.05, pos[1]-0.04), 0.1, 0.08, 
                        facecolor='#3498db', edgecolor='black', linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], f'Lab\nIndex', ha='center', va='center', 
               fontsize=9, fontweight='bold', color='white')
    
    # Disease nodes
    for i, (key, pos) in enumerate([(k, v) for k, v in positions.items() if 'disease' in k]):
        triangle = plt.Polygon([(pos[0], pos[1]+0.06), 
                               (pos[0]-0.05, pos[1]-0.04),
                               (pos[0]+0.05, pos[1]-0.04)],
                              facecolor='#e74c3c', edgecolor='black', 
                              linewidth=2, zorder=3)
        ax.add_patch(triangle)
        label = 'ICD1\n(w=1)' if i == 0 else 'ICD2\n(w=0.5)'
        ax.text(pos[0], pos[1]-0.08, label, ha='center', va='top',
               fontsize=9, fontweight='bold')
    
    # Draw edges
    # Lab edges
    for key, pos in [(k, v) for k, v in positions.items() if 'lab' in k]:
        ax.plot([positions['patient'][0], pos[0]], 
               [positions['patient'][1], pos[1]],
               color='#3498db', linewidth=2, alpha=0.7, zorder=1)
        mid_x = (positions['patient'][0] + pos[0]) / 2
        mid_y = (positions['patient'][1] + pos[1]) / 2
        ax.text(mid_x, mid_y, 'Lab\nValue', fontsize=8, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Disease edges
    ax.plot([positions['patient'][0], positions['disease1'][0]], 
           [positions['patient'][1], positions['disease1'][1]],
           color='#e74c3c', linewidth=3, alpha=0.8, zorder=1)
    
    ax.plot([positions['patient'][0], positions['disease2'][0]], 
           [positions['patient'][1], positions['disease2'][1]],
           color='#f39c12', linewidth=2, alpha=0.8, zorder=1)
    
    # Title
    ax.set_title('Heterogeneous Medical Graph Schema\n(Node Types & Edge Relations)',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add legend
    legend_text = (
        "Node Types:\n"
        "• Patient (circle): Patient records\n"
        "• Lab Index (square): Blood test features\n"
        "• Disease (triangle): ICD-10 codes\n\n"
        "Edge Relations:\n"
        "• Patient → Lab: weight = test value\n"
        "• Patient → ICD1: weight = 1 (primary)\n"
        "• Patient → ICD2: weight = 0.5 (secondary)"
    )
    
    ax.text(0.05, 0.3, legend_text, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.show()
    plt.close()

def main():
    """Main function"""
    print("="*70)
    print("MEDICAL GRAPH VISUALIZATION - SUBGRAPH EXTRACTION")
    print("="*70)
    
    # Load graph
    print("\n[1/3] Loading graph...")
    G = load_graph('data/Lab_graph_edge_list.csv')
    if G is None:
        return
    
    # Extract subgraph
    print("\n[2/3] Extracting sample subgraph...")
    subgraph = extract_sample_subgraph(G, num_patients=3, seed=42)
    
    # Visualize
    print("\n[3/3] Creating visualizations...")
    visualize_subgraph(subgraph, 'output/subgraph_sample.png')
    create_schematic_diagram('output/graph_schema.png')
    
    print("\n" + "="*70)
    print("✅ COMPLETED!")
    print("Output files:")
    print("  - output/subgraph_sample.png (Sample subgraph)")
    print("  - output/graph_schema.png (Schema diagram)")
    print("="*70)

if __name__ == "__main__":
    main()
