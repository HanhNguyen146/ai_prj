import torch
import networkx as nx
import numpy as np
from datasets import load_dataset
from torchvision import transforms, models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms import community

def run_full_lpa_pipeline(k_neighbors=5, similarity_threshold=0.5):
    # --- 1. TẢI DATASET VÀ TRÍCH XUẤT ĐẶC TRƯNG ---
    print("1. Đang tải dataset ImageNet-Hard từ Hugging Face...")
    # Thường dataset đánh giá được lưu ở split 'validation' hoặc 'test'
    dataset = load_dataset("taesiri/imagenet-hard", split="validation")
    
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1]) 
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    true_labels = [] # Lưu nhãn thực tế để đánh giá
    
    print(f"2. Đang trích xuất đặc trưng cho {len(dataset)} ảnh (có thể mất vài phút)...")
    for item in dataset:
        img = item['image'].convert('RGB')
        img_t = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            feat = model(img_t).flatten().numpy()
        
        features.append(feat)
        label_data = item['label']
        
        # Nếu nhãn là một danh sách, chỉ lấy nhãn đầu tiên
        if isinstance(label_data, list):
            true_labels.append(label_data[0]) 
        else:
            true_labels.append(label_data)
                
    # Đã đưa 2 dòng này ra ngoài vòng lặp for
    features = np.array(features)
    num_nodes = len(features)

    # --- 2. XÂY DỰNG ĐỒ THỊ K-NN TỐI ƯU ---
    print(f"3. Đang xây dựng đồ thị K-NN (K={k_neighbors})...")
    sim_matrix = cosine_similarity(features)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for i in range(num_nodes):
        top_k_indices = np.argsort(sim_matrix[i])[-k_neighbors-1:-1]
        for j in top_k_indices:
            weight = sim_matrix[i][j]
            if weight > similarity_threshold:
                G.add_edge(i, j, weight=weight)

    # --- 3. CHẠY THUẬT TOÁN LPA ---
    print("4. Đang chạy thuật toán phân cụm LPA...")
    lpa_communities = list(community.asyn_lpa_communities(G, weight='weight'))
    
    # Chuyển đổi kết quả LPA thành danh sách nhãn dự đoán để so sánh
    pred_labels = np.zeros(num_nodes)
    for cluster_id, comm in enumerate(lpa_communities):
        for node in comm:
            pred_labels[node] = cluster_id

    # --- 4. ĐÁNH GIÁ (EVALUATION) ---
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(f"Số lượng cụm tìm được: {len(lpa_communities)}")
    
    # Tính NMI và ARI
    nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
    ari_score = adjusted_rand_score(true_labels, pred_labels)
    
    print(f"Điểm NMI (Normalized Mutual Information): {nmi_score:.4f}")
    print(f"Điểm ARI (Adjusted Rand Index):         {ari_score:.4f}")
    
    return nmi_score, ari_score

# --- CHẠY THỬ ---
run_full_lpa_pipeline(k_neighbors=5, similarity_threshold=0.5)