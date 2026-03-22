import os
import torch
import networkx as nx
import numpy as np
from collections import Counter
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from networkx.algorithms import community
from transformers import CLIPProcessor, CLIPModel

# --- 1. TẢI DATA VÀ TRÍCH XUẤT (DÙNG CLIP) ---
def load_or_extract_features(cache_file="features_cache_clip.npz"): 
    if os.path.exists(cache_file):
        print("1. Đang tải đặc trưng CLIP từ file cache...")
        data = np.load(cache_file, allow_pickle=True)
        return data['features'], data['true_labels'].tolist()

    print("1. Đang tải dataset và trích xuất bằng CLIP...")
    dataset = load_dataset("taesiri/imagenet-hard", split="validation")
    
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    features, true_labels = [], []
    for item in dataset:
        img = item['image'].convert('RGB')
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad(): 
            outputs = model.get_image_features(**inputs)
            if isinstance(outputs, torch.Tensor):
                image_features = outputs.flatten().numpy()
            else:
                image_features = outputs.pooler_output.flatten().numpy()
        
        features.append(image_features)
        label_data = item['label']
        true_labels.append(label_data[0] if isinstance(label_data, list) else label_data)

    features = np.array(features)
    np.savez(cache_file, features=features, true_labels=true_labels)
    return features, true_labels

# --- 2. XÂY DỰNG ĐỒ THỊ MUTUAL K-NN VÀ CHẠY LPA ---
def run_lpa_with_params(sim_matrix, num_nodes, k, threshold):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # [NOTE: THAY ĐỔI CỐT LÕI] - Áp dụng Mutual K-NN
    # Bước A: Tìm danh sách Top K hàng xóm cho TẤT CẢ các đỉnh
    top_k_sets = []
    for i in range(num_nodes):
        # np.argsort sắp xếp tăng dần, lấy K phần tử cuối (bỏ qua chính nó ở vị trí -1)
        neighbors = np.argsort(sim_matrix[i])[-k-1:-1]
        top_k_sets.append(set(neighbors))
        
    # Bước B: Chỉ nối cạnh khi có sự đồng thuận 2 chiều
    for i in range(num_nodes):
        for j in top_k_sets[i]:
            # ĐIỀU KIỆN TƯƠNG HỖ: i phải nằm trong top K của j, và j nằm trong top K của i
            if i in top_k_sets[j]:
                weight = sim_matrix[i][j]
                if weight > threshold:
                    G.add_edge(i, j, weight=weight)
                
    # Nếu đồ thị bị đứt gãy quá nặng (ít cạnh hơn số đỉnh), bỏ qua cấu hình này
    if G.number_of_edges() < num_nodes / 2:
        return None, 0, 0

    lpa_communities = list(community.asyn_lpa_communities(G, weight='weight'))
    return lpa_communities, len(lpa_communities), G.number_of_edges() / num_nodes

# --- 3. MAIN (PCA + GRID SEARCH + XUẤT ẢNH) ---
def main():
    features, true_labels = load_or_extract_features()
    
    print("\n2. Đang nén đặc trưng bằng PCA (giảm xuống 256 chiều)...")
    pca = PCA(n_components=256, random_state=42)
    compressed_features = pca.fit_transform(features)
    
    print("\n3. Đang tính toán ma trận độ tương đồng (Cosine)...")
    sim_matrix = cosine_similarity(compressed_features)
    num_nodes = len(compressed_features)
    
    # K cần tăng lên một chút vì Mutual K-NN lọc rất gắt
    k_values = [20, 30, 50]
    threshold_values = [0.6, 0.7, 0.75] 
    
    best_ari = -1
    best_communities = None
    best_params = {}

    print("\n--- BẮT ĐẦU DÒ TÌM THAM SỐ VỚI MUTUAL K-NN ---")
    for k in k_values:
        for t in threshold_values:
            communities, num_clusters, avg_edges = run_lpa_with_params(sim_matrix, num_nodes, k, t)
            if communities is None: 
                continue
            
            pred_labels = np.zeros(num_nodes)
            for cluster_id, comm in enumerate(communities):
                for node in comm: pred_labels[node] = cluster_id
                    
            ari = adjusted_rand_score(true_labels, pred_labels)
            print(f"Thử K={k}, T={t} | Đồ thị: {avg_edges:.1f} cạnh/đỉnh | Điểm ARI: {ari:.4f} ({num_clusters} cụm)")
            
            if ari > best_ari:
                best_ari = ari
                best_communities = communities
                best_params = {'k': k, 'threshold': t}

    if best_communities is None:
        print("\nKhông tìm thấy cấu hình nào tạo ra đồ thị hợp lệ. Hãy giảm Threshold hoặc tăng K.")
        return

    print("\n--- KẾT QUẢ TỐI ƯU NHẤT ---")
    print(f"K = {best_params['k']}, Threshold = {best_params['threshold']}")
    print(f"ARI cao nhất: {best_ari:.4f}")
    
    print("\n--- CHI TIẾT TOP 5 CỤM LỚN NHẤT ĐỂ ĐÁNH GIÁ ĐỘ TINH KHIẾT ---")
    sorted_communities = sorted(best_communities, key=len, reverse=True)
    
    for i, comm in enumerate(sorted_communities[:5]):
        labels_in_cluster = [true_labels[node] for node in comm]
        top_label, top_count = Counter(labels_in_cluster).most_common(1)[0]
        purity = (top_count / len(comm)) * 100
        
        print(f"\nCụm {i+1} (chứa {len(comm)} ảnh):")
        print(f"  -> Nhãn gốc chiếm đa số: {top_label} (Độ tinh khiết: {purity:.1f}%)")
        print(f"  -> ID 5 ảnh đại diện: {list(comm)[:5]}")

    # ==========================================
    # PHẦN MỚI THÊM: XUẤT TOÀN BỘ CỤM RA THƯ MỤC 'Image'
    # ==========================================
    print("\n--- ĐANG XUẤT ẢNH RA THƯ MỤC ĐỂ TRỰC QUAN HÓA ---")
    dataset = load_dataset("taesiri/imagenet-hard", split="validation")
    
    # 1. Tạo thư mục cha tên là 'Image'
    main_output_dir = "Image"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # 2. Lấy số lượng cụm (có thể giới hạn lại nếu chạy quá lâu, vd: 50 hoặc 100)
    num_clusters_to_export = len(sorted_communities) 
    print(f"Đang lưu {num_clusters_to_export} cụm vào thư mục '{main_output_dir}' (việc này có thể mất vài phút)...")
    
    for i, comm in enumerate(sorted_communities[:num_clusters_to_export]):
        # Tạo thư mục con cho từng cụm (vd: Image/Cluster_1)
        cluster_dir = os.path.join(main_output_dir, f"Cluster_{i+1}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Lưu tất cả ảnh của cụm đó vào thư mục con
        for img_id in comm:
            img = dataset[img_id]['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(os.path.join(cluster_dir, f"img_ID{img_id}.jpg"))
            
        # In tiến độ để tiện theo dõi
        if (i + 1) % 100 == 0:
            print(f"  -> Đã xuất xong {i + 1} cụm...")
            
    print(f"\n-> HOÀN TẤT! Toàn bộ ảnh đã được phân loại vào thư mục '{main_output_dir}'.")

if __name__ == "__main__":
    main()