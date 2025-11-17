import pandas as pd
import networkx as nx
import numpy as np

# --- Cấu hình ---
file_path = "data/lab_with_icd10.csv"
# Các cột chứa chỉ số xét nghiệm (sẽ là trọng số cạnh lab)
lab_columns = [
    'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC',
    'PLT', 'WBC', 'NEUT', 'EO', 'BASO', 'MONO', 'LYMPH',
    'MPV', 'PCT', 'PDW'
]
patient_id_column = 'Patient_ID'
icd1_column = 'ICD1'
icd2_column = 'ICD2' # Cột ICD2 chứa các mã bệnh cần tách

# --- 1. Xử lý dữ liệu ---
try:
    df = pd.read_csv(file_path)

    # Đảm bảo cột Patient_ID là chuỗi (string)
    df[patient_id_column] = df[patient_id_column].astype(str)

    # Xử lý các cột ICD1 và ICD2 bằng cách điền NaN bằng chuỗi 'None'
    df[icd1_column] = df[icd1_column].fillna('None').astype(str)
    df[icd2_column] = df[icd2_column].fillna('None').astype(str)

    # Đảm bảo các cột xét nghiệm là dạng số, điền NaN bằng 0
    for col in lab_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp tại đường dẫn: {file_path}")
    exit()

# --- 2. Xây dựng Biểu đồ (Graph) ---
G = nx.Graph()

# 2.1. Thêm các Đỉnh (Nodes)
patient_nodes = df[patient_id_column].unique().tolist()
G.add_nodes_from(patient_nodes, type='Patient')
G.add_nodes_from(lab_columns, type='Lab_Index')

# --- 3. Thêm các Cạnh (Edges) và Trọng số (Weights) ---
print("Đang thêm cạnh và trọng số theo quy tắc đã định...")

# Duyệt qua từng hàng dữ liệu (từng lần khám của bệnh nhân)
for index, row in df.iterrows():
    patient = row[patient_id_column]

    # --- a) Cạnh Bệnh nhân <--> Chỉ số xét nghiệm (Trọng số là Giá trị xét nghiệm) ---
    for lab_index in lab_columns:
        weight = row[lab_index]
        if weight != 0:
            # Gán trọng số là giá trị xét nghiệm
            G.add_edge(patient, lab_index, weight=weight, relation='Lab_Value')

    # --- b) Cạnh Bệnh nhân <--> ICD1 (Trọng số cố định = 1) ---
    icd1_code = row[icd1_column]
    if icd1_code != 'None':
        # Thêm đỉnh ICD1 nếu chưa có
        if icd1_code not in G:
             G.add_node(icd1_code, type='Disease')
        
        # Gán trọng số cố định = 1
        G.add_edge(patient, icd1_code, weight=1, relation='Have_Disease')

    # --- c) Cạnh Bệnh nhân <--> ICD2 (Tách mã, Trọng số cố định = 2) ---
    icd2_codes_raw = row[icd2_column]
    
    # Giả sử các mã ICD2 được phân tách bằng dấu chấm phẩy (;) hoặc dấu phẩy (,)
    # Ở đây ta giả định dựa trên snippet dữ liệu, ICD2 là một mã duy nhất hoặc NaN
    # Nhưng để đáp ứng yêu cầu "ICD2 có nhiều mã bệnh nên cần tách riêng", ta sử dụng regex hoặc split
    
    # Đối với dữ liệu CSV thường thấy, cột ICD2 có thể chứa một chuỗi mã bệnh cách nhau bởi dấu ',' hoặc ';'
    if icd2_codes_raw != 'None':
        # Giả định tách các mã bằng dấu phẩy hoặc khoảng trắng (cần điều chỉnh theo định dạng thực tế)
        icd2_codes = [code.strip() for code in str(icd2_codes_raw).split(';') if code.strip()]
        
        for icd2_code in icd2_codes:
            # Kiểm tra và chỉ thêm nếu mã ICD2 không trùng với ICD1
            if icd2_code != icd1_code and icd2_code != 'None':
                
                # Thêm đỉnh ICD2 nếu chưa có
                if icd2_code not in G:
                     G.add_node(icd2_code, type='Disease')
                
                # Gán trọng số cố định = 0.5
                G.add_edge(patient, icd2_code, weight=0.5, relation='Have_Disease')


# --- 4. Tổng kết ---
print("\n✅ Biểu đồ (Graph) đã được xây dựng thành công.")
print("---")
print(f"Tổng số Đỉnh (Nodes): {G.number_of_nodes()}")
print(f"Tổng số Cạnh (Edges): {G.number_of_edges()}")
print("---")
print("Cấu trúc trọng số:")
print("- Cạnh Bệnh nhân <--> Lab Index: Trọng số = Giá trị xét nghiệm")
print("- Cạnh Bệnh nhân <--> ICD1: Trọng số = 1 (Chẩn đoán chính)")
print("- Cạnh Bệnh nhân <--> ICD2: Trọng số = 0.5 (Chẩn đoán phụ)")

# Lưu đồ thị thành file .csv dạng chuẩn edge list
output_edge_list_path = "data/Lab_graph_edge_list.csv"
nx.write_edgelist(G, output_edge_list_path, data=['weight', 'relation'])
print(f"Đã lưu danh sách cạnh vào file: {output_edge_list_path}")   

# Lưu đồ thị dưới dạng file GEXF để sử dụng trong Gephi hoặc các công cụ khác
output_gexf_path = "output/Lab_graph.gexf"
nx.write_gexf(G, output_gexf_path)
print(f"Đã lưu đồ thị vào file: {output_gexf_path}")