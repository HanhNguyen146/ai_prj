import os
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
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

# --- 2. MAIN (PCA + GMM) ---
def main():
    features, true_labels = load_or_extract_features()
    
    print("\n2. Đang nén đặc trưng bằng PCA (giảm xuống 256 chiều)...")
    pca = PCA(n_components=256, random_state=42)
    compressed_features = pca.fit_transform(features)
    
    # GMM tự tính toán phương sai nên không cần dùng hàm normalize() như K-Means
    k_components_list = [800, 1000, 1200]
    
    best_ari = -1
    best_labels = None
    best_k = 0

    print("\n--- BẮT ĐẦU CHẠY THUẬT TOÁN GAUSSIAN MIXTURE MODEL (GMM) ---")
    for k_clusters in k_components_list:
        print(f"Đang chạy GMM với {k_clusters} cụm (việc này có thể mất vài phút)...")
        
        # covariance_type='diag' giúp tính toán nhanh và phù hợp với dữ liệu nhiều chiều
        gmm = GaussianMixture(n_components=k_clusters, covariance_type='diag', random_state=42, max_iter=100)
        pred_labels = gmm.fit_predict(compressed_features)
        
        ari = adjusted_rand_score(true_labels, pred_labels)
        print(f"-> Thử K={k_clusters} cụm | Điểm ARI: {ari:.4f}")
        
        if ari > best_ari:
            best_ari = ari
            best_labels = pred_labels
            best_k = k_clusters

    print("\n--- KẾT QUẢ TỐI ƯU NHẤT VỚI GMM ---")
    print(f"Số lượng cụm tối ưu: {best_k}")
    print(f"ARI cao nhất: {best_ari:.4f}")
    
    # Gom nhóm kết quả lại để đánh giá Purity và xuất ảnh
    clusters = {}
    for node_idx, cluster_id in enumerate(best_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node_idx)
        
    communities = list(clusters.values())
    sorted_communities = sorted(communities, key=len, reverse=True)
    
    print("\n--- CHI TIẾT TOP 5 CỤM LỚN NHẤT ---")
    for i, comm in enumerate(sorted_communities[:5]):
        labels_in_cluster = [true_labels[node] for node in comm]
        top_label, top_count = Counter(labels_in_cluster).most_common(1)[0]
        purity = (top_count / len(comm)) * 100
        print(f"\nCụm {i+1} (chứa {len(comm)} ảnh):")
        print(f"  -> Nhãn gốc chiếm đa số: {top_label} (Độ tinh khiết: {purity:.1f}%)")
        print(f"  -> ID 5 ảnh đại diện: {list(comm)[:5]}")

    # ==========================================
    # XUẤT ẢNH RA THƯ MỤC
    # ==========================================
    print("\n--- ĐANG XUẤT ẢNH RA THƯ MỤC ĐỂ TRỰC QUAN HÓA ---")
    dataset = load_dataset("taesiri/imagenet-hard", split="validation")
    main_output_dir = "Image_GMM"
    os.makedirs(main_output_dir, exist_ok=True)
    
    num_clusters_to_export = min(100, len(sorted_communities))
    print(f"Đang lưu {num_clusters_to_export} cụm lớn nhất vào thư mục '{main_output_dir}'...")
    
    for i, comm in enumerate(sorted_communities[:num_clusters_to_export]):
        cluster_dir = os.path.join(main_output_dir, f"Cluster_{i+1}")
        os.makedirs(cluster_dir, exist_ok=True)
        for img_id in comm:
            img = dataset[img_id]['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(os.path.join(cluster_dir, f"img_ID{img_id}.jpg"))
            
        if (i + 1) % 25 == 0:
            print(f"  -> Đã xuất xong {i + 1} cụm...")
            
    print(f"\n-> HOÀN TẤT! Ảnh đã được lưu vào '{main_output_dir}'.")

if __name__ == "__main__":
    main()