# gnn_diag_minimal.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# ===========================
# 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ===========================

def load_graph_data(filepath='data/Lab_graph_edge_list.csv'):
    """
    Đọc dữ liệu đồ thị từ file CSV
    Format: source | target | weight | edge_type
    """
    df = pd.read_csv(filepath, sep=' ', header=None, 
                     names=['source', 'target', 'weight', 'edge_type'])
    return df

def create_hetero_patient_graphs(df):
    """
    Tạo Heterogeneous Graph cho từng bệnh nhân
    3 loại node:
    - Patient: 1 node (bệnh nhân)
    - Lab: 16 nodes (các chỉ số xét nghiệm)
    - Disease: M nodes (TẤT CẢ các bệnh)
    Trọng số cạnh (patient -> disease) lấy từ cột 'weight'
    """
    lab_features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'WBC', 
                    'NEUT', 'EO', 'BASO', 'MONO', 'LYMPH', 'MPV', 'PCT', 'PDW']
    
    print(f"Edge types trong dữ liệu: {df['edge_type'].unique()}")
    disease_edges = df[df['edge_type'] == 'Have_Disease']
    print(f"Số dòng Have_Disease: {len(disease_edges)}")
    
    all_diseases = sorted(disease_edges['target'].unique())
    disease_to_idx = {code: idx for idx, code in enumerate(all_diseases)}
    # Sử dụng chung disease_to_idx cho cả label và node mapping
    icd_to_idx = disease_to_idx
    
    print(f"Tổng số bệnh: {len(all_diseases)}")
    if len(all_diseases) > 0:
        print(f"Ví dụ 5 bệnh đầu: {all_diseases[:5]}")
    
    lab_to_idx = {feat: idx for idx, feat in enumerate(lab_features)}
    
    graphs = []
    labels = []
    patient_ids_list = []
    
    unique_patients = df['source'].unique()
    print(f"Số lượng patients: {len(unique_patients)}")
    
    for patient_id in unique_patients:
        patient_data = df[df['source'] == patient_id]
        patient_diseases = patient_data[patient_data['edge_type'] == 'Have_Disease']
        if len(patient_diseases) == 0:
            continue
        
        primary_disease = patient_diseases.iloc[0]['target']
        if primary_disease not in icd_to_idx:
            print(f"Warning: Disease {primary_disease} not in icd_to_idx")
            continue
        label = icd_to_idx[primary_disease]
        
        hetero_data = HeteroData()
        hetero_data['patient'].x = torch.zeros(1, 16)
        hetero_data['patient'].y = torch.tensor([label], dtype=torch.long)
        
        lab_data = patient_data[patient_data['edge_type'] == 'Have_Lab']
        lab_node_features = torch.zeros(len(lab_features), 1)
        for _, row in lab_data.iterrows():
            feat_name = row['target']
            if feat_name in lab_to_idx:
                idx = lab_to_idx[feat_name]
                lab_node_features[idx, 0] = float(row['weight'])
        hetero_data['lab'].x = lab_node_features
        
        disease_list = []
        disease_weights = []
        for _, row in patient_diseases.iterrows():
            disease_code = row['target']
            if disease_code in disease_to_idx:
                disease_list.append(disease_to_idx[disease_code])
                try:
                    weight = float(row['weight'])
                except:
                    weight = 1.0
                disease_weights.append(weight)
        
        num_diseases = max(len(disease_list), 1)
        disease_node_features = torch.zeros(num_diseases, 2)
        for idx, (disease_idx, weight) in enumerate(zip(disease_list, disease_weights)):
            disease_node_features[idx, 0] = disease_idx / len(all_diseases)
            disease_node_features[idx, 1] = weight
        hetero_data['disease'].x = disease_node_features
        
        patient_lab_edges = torch.tensor([[0] * len(lab_features), 
                                          list(range(len(lab_features)))], dtype=torch.long)
        hetero_data['patient', 'has_lab', 'lab'].edge_index = patient_lab_edges
        
        patient_disease_edges = torch.tensor([[0] * num_diseases, 
                                             list(range(num_diseases))], dtype=torch.long)
        hetero_data['patient', 'has_disease', 'disease'].edge_index = patient_disease_edges
        hetero_data['patient', 'has_disease', 'disease'].edge_attr = torch.tensor(
            disease_weights, dtype=torch.float).unsqueeze(1)
        
        hetero_data['lab', 'rev_has_lab', 'patient'].edge_index = patient_lab_edges.flip([0])
        hetero_data['disease', 'rev_has_disease', 'patient'].edge_index = patient_disease_edges.flip([0])
        hetero_data['disease', 'rev_has_disease', 'patient'].edge_attr = torch.tensor(
            disease_weights, dtype=torch.float).unsqueeze(1)
        
        graphs.append(hetero_data)
        labels.append(label)
        patient_ids_list.append(patient_id)
    
    print(f"Đã tạo {len(graphs)} graphs")
    
    return graphs, labels, patient_ids_list, icd_to_idx, lab_to_idx, disease_to_idx

# ===========================
# 2. ĐỊNH NGHĨA MÔ HÌNH HETERO GNN
# ===========================

class HeteroGNN_Diagnosis(nn.Module):
    """
    Heterogeneous Graph Neural Network cho chẩn đoán bệnh
    3 loại node: Patient, Lab, Disease
    """
    def __init__(self, hidden_dim, output_dim, dropout=0.5):
        super(HeteroGNN_Diagnosis, self).__init__()
        
        # Input projections
        self.patient_lin = Linear(16, hidden_dim)
        self.lab_lin = Linear(1, hidden_dim)
        self.disease_lin = Linear(2, hidden_dim)
        
        # Heterogeneous Graph Convolution Layers
        self.conv1 = HeteroConv({
            ('patient', 'has_lab', 'lab'): SAGEConv(hidden_dim, hidden_dim),
            ('patient', 'has_disease', 'disease'): SAGEConv(hidden_dim, hidden_dim),
            ('lab', 'rev_has_lab', 'patient'): SAGEConv(hidden_dim, hidden_dim),
            ('disease', 'rev_has_disease', 'patient'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('patient', 'has_lab', 'lab'): SAGEConv(hidden_dim, hidden_dim),
            ('patient', 'has_disease', 'disease'): SAGEConv(hidden_dim, hidden_dim),
            ('lab', 'rev_has_lab', 'patient'): SAGEConv(hidden_dim, hidden_dim),
            ('disease', 'rev_has_disease', 'patient'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='sum')
        
        # Batch normalization
        self.bn1 = nn.ModuleDict({
            'patient': nn.BatchNorm1d(hidden_dim),
            'lab': nn.BatchNorm1d(hidden_dim),
            'disease': nn.BatchNorm1d(hidden_dim)
        })
        
        self.bn2 = nn.ModuleDict({
            'patient': nn.BatchNorm1d(hidden_dim),
            'lab': nn.BatchNorm1d(hidden_dim),
            'disease': nn.BatchNorm1d(hidden_dim)
        })
        
        # Classification head
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, output_dim)
        
        self.dropout = dropout
    
    def forward(self, data):
        # Project input features
        x_dict = {
            'patient': self.patient_lin(data['patient'].x),
            'lab': self.lab_lin(data['lab'].x),
            'disease': self.disease_lin(data['disease'].x)
        }
        
        # Layer 1
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: self.bn1[key](x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Layer 2
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {key: self.bn2[key](x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Patient node embedding cho classification
        patient_emb = x_dict['patient']
        
        # Classification
        x = F.relu(self.fc1(patient_emb))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# ===========================
# 3. TRAINING VÀ EVALUATION
# ===========================

def train(model, loader, optimizer, criterion, device):
    """Training loop"""
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        # Fix: Lấy y từ patient node
        labels = data['patient'].y
        loss = criterion(out, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total_samples += labels.size(0)
    
    return total_loss / total_samples, correct / total_samples

def evaluate(model, loader, criterion, device):
    """Evaluation loop"""
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    total_samples = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            # Fix: Lấy y từ patient node
            labels = data['patient'].y
            loss = criterion(out, labels)
            
            total_loss += loss.item() * labels.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / total_samples, correct / total_samples, all_preds, all_labels

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Vẽ biểu đồ loss và accuracy"""
    os.makedirs('output', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(val_losses, label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(train_accs, label='Train Acc', linewidth=2, marker='o', markersize=4)
    ax2.plot(val_accs, label='Val Acc', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/training_history.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: output/training_history.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, icd_to_idx):
    """Vẽ confusion matrix"""
    os.makedirs('output', exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(icd_to_idx.keys()),
                yticklabels=sorted(icd_to_idx.keys()),
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: output/confusion_matrix.png")
    plt.close()

# ===========================
# 4. CHẠY TOÀN BỘ QUÁ TRÌNH
# ===========================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dữ liệu
    print("Đang load dữ liệu...")
    df = load_graph_data('data/Lab_graph_edge_list.csv')
    
    # DEBUG: In thông tin về dataframe
    print(f"\nShape của dataframe: {df.shape}")
    print(f"Các cột: {df.columns.tolist()}")
    print(f"\n5 dòng đầu:")
    print(df.head())
    
    graphs, labels, patient_ids_list, icd_to_idx, lab_to_idx, disease_to_idx = create_hetero_patient_graphs(df)
    
    # Kiểm tra số lượng graphs
    if len(graphs) == 0:
        print("\n❌ ERROR: Không có graph nào được tạo!")
        print("Kiểm tra:")
        print("1. Edge_type 'Have_Disease' có tồn tại không?")
        print("2. Edge_type 'Have_Lab' có tồn tại không?")
        print("3. Dữ liệu có đúng format: source | target | weight | edge_type không?")
        exit(1)
    
    print(f"\n✓ Đã tạo {len(graphs)} graphs thành công!")
    
    # ===== LỌC BỎ CÁC CLASS CÓ ÍT HƠN 2 SAMPLES =====
    label_counts = Counter(labels)
    
    print(f"\n📊 Phân phối class trước khi lọc:")
    for disease_code, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        disease_name = [k for k, v in icd_to_idx.items() if v == disease_code][0]
        print(f"  {disease_name}: {count} samples")
    
    # Lọc bỏ class có < 2 samples
    min_samples = 2
    valid_indices = []
    for i, label in enumerate(labels):
        if label_counts[label] >= min_samples:
            valid_indices.append(i)
    
    # Cập nhật danh sách
    filtered_graphs = [graphs[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    filtered_patient_ids = [patient_ids_list[i] for i in valid_indices]
    
    # Tạo lại icd_to_idx chỉ với các class còn lại
    unique_filtered_labels = sorted(set(filtered_labels))
    old_to_new_label = {old_label: new_label for new_label, old_label in enumerate(unique_filtered_labels)}
    
    # Remap labels sang 0, 1, 2, ...
    remapped_labels = [old_to_new_label[label] for label in filtered_labels]
    
    # Cập nhật y trong graphs
    for graph, new_label in zip(filtered_graphs, remapped_labels):
        graph['patient'].y = torch.tensor([new_label], dtype=torch.long)
    
    # Tạo lại icd_to_idx mapping
    idx_to_disease_old = {v: k for k, v in icd_to_idx.items()}
    new_icd_to_idx = {}
    for new_idx, old_idx in enumerate(unique_filtered_labels):
        disease_code = idx_to_disease_old[old_idx]
        new_icd_to_idx[disease_code] = new_idx
    
    print(f"\n🔍 Sau khi lọc:")
    print(f"  Loại bỏ: {len(graphs) - len(filtered_graphs)} samples")
    print(f"  Còn lại: {len(filtered_graphs)} samples với {len(new_icd_to_idx)} classes")
    
    print(f"\n📊 Phân phối class sau khi lọc:")
    remapped_label_counts = Counter(remapped_labels)
    for new_label, count in sorted(remapped_label_counts.items()):
        disease_code = [k for k, v in new_icd_to_idx.items() if v == new_label][0]
        print(f"  {disease_code}: {count} samples")
    
    # Kiểm tra lại
    if len(filtered_graphs) < 10:
        print(f"\n❌ ERROR: Quá ít samples ({len(filtered_graphs)}) sau khi lọc!")
        exit(1)
    
    # Sử dụng filtered data
    graphs = filtered_graphs
    labels = remapped_labels
    patient_ids_list = filtered_patient_ids
    icd_to_idx = new_icd_to_idx
    
    # Chia tập train/val/test
    try:
        train_graphs, test_graphs, train_labels, test_labels = train_test_split(
            graphs, labels, test_size=0.2, random_state=42, stratify=labels)
        train_graphs, val_graphs, train_labels, val_labels = train_test_split(
            train_graphs, train_labels, test_size=0.1, random_state=42, stratify=train_labels)
    except ValueError as e:
        print(f"\n⚠️  Không thể stratified split: {e}")
        print("  Sử dụng random split thay thế...")
        train_graphs, test_graphs, train_labels, test_labels = train_test_split(
            graphs, labels, test_size=0.2, random_state=42)
        train_graphs, val_graphs, train_labels, val_labels = train_test_split(
            train_graphs, train_labels, test_size=0.1, random_state=42)
    
    print(f"\nSố lượng graph - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Tạo DataLoader
    batch_size = 32
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    # Khởi tạo mô hình
    hidden_dim = 128
    output_dim = len(icd_to_idx)
    model = HeteroGNN_Diagnosis(hidden_dim, output_dim, dropout=0.3).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.NLLLoss()
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    patience = 50  # Số epoch không cải thiện trước khi dừng
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)    
        
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'icd_to_idx': icd_to_idx,
                'lab_to_idx': lab_to_idx,
                'disease_to_idx': disease_to_idx,
                'num_classes': output_dim,
                'hidden_dim': hidden_dim,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs
            }, 'output/best_hetero_model.pth')
            print("✓ Model saved!")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping: Không cải thiện val_acc sau {patience} epoch liên tiếp.")
            break

    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, icd_to_idx)