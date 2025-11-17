"""
Script để chẩn đoán bệnh nhân mới CHỈ TỪ KẾT QUẢ XÉT NGHIỆM MÁU
Sử dụng Heterogeneous Graph + xAI (Explainable AI)
"""

import torch
from Lab_GNN import (
    load_graph_data, 
    create_hetero_patient_graphs,
    HeteroGNN_Diagnosis
)
from torch_geometric.data import HeteroData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_default_values_multi():
    """
    Hồ sơ nhiều bệnh gợi ý: viêm phổi/nhiễm trùng (WBC↑, NEUT↑),
    thiếu máu thiếu sắt (HGB↓, MCV↓, MCH↓, MCHC↓), tiểu cầu phản ứng (PLT↑),
    kèm tăng nhẹ eosinophil.
    """
    return {
        'RBC': 3.9,     # thấp nhẹ
        'HGB': 95,      # thiếu máu
        'HCT': 0.31,    # theo HGB thấp
        'MCV': 68,      # microcytic
        'MCH': 20.5,    # thấp
        'MCHC': 300,    # thấp
        'PLT': 480,     # thrombocytosis phản ứng
        'WBC': 15.0,    # bạch cầu cao
        'NEUT': 78.0,   # tăng trung tính
        'EO': 3.0,      # eosinophil hơi cao
        'BASO': 0.5,
        'MONO': 9.0,
        'LYMPH': 15.0,  # lympho tương đối thấp
        'MPV': 7.6,
        'PCT': 0.45,
        'PDW': 45.0
    }

def create_patient_graph_from_lab(lab_values, lab_to_idx, disease_to_idx):
    """
    Tạo HeteroData graph cho bệnh nhân mới CHỈ TỪ LAB VALUES
    """
    hetero_data = HeteroData()
    
    # Patient node
    hetero_data['patient'].x = torch.zeros(1, 16)
    
    # Lab nodes
    lab_features = list(lab_to_idx.keys())
    lab_node_features = torch.zeros(len(lab_features), 1)
    for lab_name, value in lab_values.items():
        if lab_name in lab_to_idx:
            idx = lab_to_idx[lab_name]
            lab_node_features[idx, 0] = value
    hetero_data['lab'].x = lab_node_features
    
    # Disease nodes (placeholder - model sẽ tự predict)
    hetero_data['disease'].x = torch.zeros(1, 2)
    
    # Edges
    patient_lab_edges = torch.tensor([[0] * len(lab_features), 
                                      list(range(len(lab_features)))], dtype=torch.long)
    hetero_data['patient', 'has_lab', 'lab'].edge_index = patient_lab_edges
    hetero_data['lab', 'rev_has_lab', 'patient'].edge_index = patient_lab_edges.flip([0])
    
    patient_disease_edges = torch.tensor([[0], [0]], dtype=torch.long)
    hetero_data['patient', 'has_disease', 'disease'].edge_index = patient_disease_edges
    hetero_data['disease', 'rev_has_disease', 'patient'].edge_index = patient_disease_edges.flip([0])
    
    return hetero_data

def compute_feature_importance_integrated_gradients(model, patient_graph, lab_to_idx, device, steps=50):
    """
    Tính độ quan trọng của từng lab feature bằng Integrated Gradients
    """
    model.eval()
    patient_graph = patient_graph.to(device)
    
    # Baseline: all zeros
    baseline_graph = patient_graph.clone()
    baseline_graph['lab'].x = torch.zeros_like(baseline_graph['lab'].x)
    
    # Get predicted class
    with torch.no_grad():
        output = model(patient_graph)
        pred_class = output.argmax(dim=1).item()
    
    importances = {}
    lab_features = list(lab_to_idx.keys())
    
    for i, lab_name in enumerate(lab_features):
        # Integrated Gradients
        gradients = []
        for alpha in torch.linspace(0, 1, steps).to(device):
            interpolated_graph = patient_graph.clone()
            interpolated_graph['lab'].x = baseline_graph['lab'].x + alpha * (
                patient_graph['lab'].x - baseline_graph['lab'].x
            )
            interpolated_graph['lab'].x.requires_grad = True
            
            # Forward pass
            output = model(interpolated_graph)
            
            # Backward
            model.zero_grad()
            output[0, pred_class].backward(retain_graph=True)
            
            if interpolated_graph['lab'].x.grad is not None:
                gradients.append(interpolated_graph['lab'].x.grad[i].cpu().detach())
        
        if gradients:
            avg_gradient = torch.stack(gradients).mean(dim=0)
            feature_diff = (patient_graph['lab'].x[i] - baseline_graph['lab'].x[i]).cpu()
            importance = (avg_gradient * feature_diff).sum().item()
            importances[lab_name] = abs(importance)
    
    # Normalize
    total = sum(importances.values()) if importances else 1.0
    if total > 0:
        importances = {k: v/total for k, v in importances.items()}
    
    return importances, pred_class

def visualize_feature_importance(importances, lab_values, top_n=10):
    """
    Vẽ biểu đồ các lab features quan trọng nhất
    """
    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, scores = zip(*sorted_features)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Bar chart - Feature Importance
    colors = plt.cm.RdYlGn(np.array(scores) / max(scores))
    bars = ax1.barh(range(len(features)), scores, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(features, fontsize=10)
    ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Most Important Lab Features', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax1.text(score, i, f' {score*100:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    # 2. Heatmap - Lab values vs Importance
    lab_data = []
    importance_data = []
    for feat in features:
        lab_data.append(lab_values.get(feat, 0))
        importance_data.append(importances[feat])
    
    # Normalize lab values for visualization
    lab_data_norm = (np.array(lab_data) - np.min(lab_data)) / (np.max(lab_data) - np.min(lab_data) + 1e-6)
    
    data_matrix = np.column_stack([lab_data_norm, importance_data])
    sns.heatmap(data_matrix, annot=[[f'{v:.2f}', f'{i*100:.1f}%'] 
                                     for v, i in zip(lab_data, importance_data)],
                fmt='', cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Score'},
                yticklabels=features, xticklabels=['Lab Value\n(normalized)', 'Importance\n(%)'])
    ax2.set_title('Lab Values vs Feature Importance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/feature_importance.png', dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: output/feature_importance.png")
    return fig

def generate_explanation_text(importances, results, lab_values):
    """
    Tạo văn bản giải thích dễ hiểu cho bác sĩ
    """
    predicted_disease = results['predicted_class']
    confidence = results['confidence']
    
    explanation = []
    explanation.append(f"\n{'='*70}")
    explanation.append(f"GIẢI THÍCH KẾT QUẢ CHẨN ĐOÁN (Explainable AI)")
    explanation.append(f"{'='*70}\n")
    
    # Main prediction
    explanation.append(f"🔍 Chẩn đoán: {predicted_disease}")
    explanation.append(f"   Độ tin cậy: {confidence*100:.2f}%\n")
    
    # Confidence interpretation
    if confidence > 0.8:
        conf_text = "RẤT CAO - Model rất chắc chắn"
    elif confidence > 0.6:
        conf_text = "CAO - Model khá chắc chắn"
    elif confidence > 0.4:
        conf_text = "TRUNG BÌNH - Cần cân nhắc thêm"
    else:
        conf_text = "THẤP - Cần thêm xét nghiệm"
    explanation.append(f"   Đánh giá: {conf_text}\n")
    
    # Top important features
    explanation.append(f"📊 CÁC CHỈ SỐ XÉT NGHIỆM QUAN TRỌNG NHẤT:\n")
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for i, (feature, importance) in enumerate(sorted_features, 1):
        bar = "█" * int(importance * 20)
        value = lab_values.get(feature, 0)
        explanation.append(f"   {i}. {feature:8s}: {bar} ({importance*100:.1f}%) | Giá trị: {value}")
    
    explanation.append(f"\n   → Các chỉ số này có ảnh hưởng lớn nhất đến chẩn đoán.\n")
    
    # Clinical interpretation
    explanation.append(f"🩺 DIỄN GIẢI LÂM SÀNG:\n")
    for feature, importance in sorted_features[:3]:
        value = lab_values.get(feature, 0)
        if feature == 'WBC' and value > 11:
            explanation.append(f"   • WBC cao ({value}) → Nhiễm trùng/Viêm")
        elif feature == 'HGB' and value < 120:
            explanation.append(f"   • HGB thấp ({value}) → Thiếu máu")
        elif feature == 'PLT' and value > 400:
            explanation.append(f"   • PLT cao ({value}) → Tiểu cầu phản ứng")
        elif feature == 'NEUT' and value > 70:
            explanation.append(f"   • NEUT cao ({value}%) → Nhiễm khuẩn")
    
    explanation.append(f"\n{'='*70}\n")
    
    return "\n".join(explanation)

def main():
    print("\n" + "="*70)
    print("CHẨN ĐOÁN BỆNH NHÂN MỚI VỚI GIẢI THÍCH (xAI)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model và mappings
    print("\n[1/5] Đang load model...")
    try:
        checkpoint = torch.load('best_hetero_model.pth', map_location=device)
        icd_to_idx = checkpoint['icd_to_idx']
        lab_to_idx = checkpoint['lab_to_idx']
        disease_to_idx = checkpoint['disease_to_idx']
        hidden_dim = checkpoint['hidden_dim']
        output_dim = checkpoint['num_classes']
        
        model = HeteroGNN_Diagnosis(hidden_dim, output_dim, dropout=0.3).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded! Classes: {output_dim}")
        
    except FileNotFoundError:
        print("✗ File 'best_hetero_model.pth' không tồn tại!")
        print("👉 Chạy: python Lab_GNN.py")
        return
    except KeyError:
        print("✗ Model cũ thiếu mappings!")
        print("👉 Train lại: python Lab_GNN.py")
        return
    
    # Tính lab statistics
    print("\n[2/5] Đang tính statistics...")
    df = load_graph_data('data/Lab_graph_edge_list.csv')
    lab_features = list(lab_to_idx.keys())
    all_lab_values = {feat: [] for feat in lab_features}
    
    sample_patients = df['source'].unique()[:100]
    for patient_id in sample_patients:
        patient_data = df[df['source'] == patient_id]
        lab_data = patient_data[patient_data['edge_type'] == 'Have_Lab']
        for _, row in lab_data.iterrows():
            feat_name = row['target']
            if feat_name in all_lab_values:
                try:
                    all_lab_values[feat_name].append(float(row['weight']))
                except:
                    pass
    
    lab_stats = {}
    for feat in lab_features:
        if len(all_lab_values[feat]) > 0:
            lab_stats[feat] = {
                'mean': np.mean(all_lab_values[feat]),
                'std': np.std(all_lab_values[feat]) + 1e-6
            }
        else:
            lab_stats[feat] = {'mean': 0, 'std': 1}
    
    print(f"✓ Statistics ready")
    
    # Patient info
    print("\n[3/5] Thông tin bệnh nhân...")
    
    patient_info = {
        'ID': "NEW_PATIENT_DEMO",
        'Tuổi': "45",
        'Giới tính': "Nam",
        'Triệu chứng': "Sốt cao, ho, khó thở"
    }
    
    print(f"Patient ID: {patient_info['ID']}")
    print(f"Tuổi: {patient_info['Tuổi']}")
    print(f"Giới tính: {patient_info['Giới tính']}")
    print(f"Triệu chứng: {patient_info['Triệu chứng']}")
    
    # Sử dụng giá trị xét nghiệm mặc định (multi-disease profile)
    print("\nSử dụng kết quả xét nghiệm máu mặc định (đa bệnh: nhiễm trùng + thiếu máu thiếu sắt + TC phản ứng):")
    default_values = make_default_values_multi()
    
    lab_values = {}
    for lab_name in lab_features:
        lab_values[lab_name] = default_values.get(lab_name, 0)
    
    # Hiển thị các giá trị xét nghiệm quan trọng
    print("Các chỉ số chính:")
    important_labs = ['WBC', 'NEUT', 'LYMPH', 'RBC', 'HGB', 'PLT']
    for lab_name in important_labs:
        if lab_name in lab_values:
            print(f"  {lab_name}: {lab_values[lab_name]}")
    print(f"  ... (và {len(lab_values) - len(important_labs)} chỉ số khác)")
    
    # Diagnosis với xAI
    print("\n[4/5] Đang chẩn đoán VỚI GIẢI THÍCH (xAI)...")
    patient_graph = create_patient_graph_from_lab(lab_values, lab_to_idx, disease_to_idx)
    patient_graph = patient_graph.to(device)
    
    model.eval()
    with torch.no_grad():
        log_probs = model(patient_graph)
        probs = torch.exp(log_probs).squeeze(0)
        
        # Top 5 predictions
        top_probs, top_indices = torch.topk(probs, k=5)
        idx_to_icd = {idx: code for code, idx in icd_to_idx.items()}
        
        print("\nKẾT QUẢ CHẨN ĐOÁN:")
        print("-"*70)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
            disease = idx_to_icd[int(idx)]
            print(f"{i}. {disease}: {float(prob)*100:.2f}%")
        print("-"*70)
        
        results = {
            'predicted_class': idx_to_icd[int(top_indices[0])],
            'confidence': float(top_probs[0])
        }
    
    # xAI - Feature Importance
    print("\n[5/5] Đang tính toán độ quan trọng của các chỉ số (xAI)...")
    importances, pred_class = compute_feature_importance_integrated_gradients(
        model, patient_graph, lab_to_idx, device, steps=30
    )
    
    # Generate explanation
    explanation_text = generate_explanation_text(importances, results, lab_values)
    print(explanation_text)
    
    # Visualize
    import os
    os.makedirs('output', exist_ok=True)
    visualize_feature_importance(importances, lab_values, top_n=10)
    
    print("\n✅ Hoàn tất chẩn đoán với giải thích!")

if __name__ == "__main__":
    main()
