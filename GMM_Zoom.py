import os
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# --- 1. TẢI DATA VÀ TRÍCH XUẤT ĐẶC TRƯNG (CẢI TIẾN VỚI CHIẾN THUẬT ZOOM) ---
# Tên file cache mới để không bị ghi đè lên dữ liệu cũ chưa Zoom
def load_or_extract_features(cache_file="features_cache_zoomed.npz"): 
    if os.path.exists(cache_file):
        print("1. Đang tải đặc trưng CLIP (Đã Zoom) từ file cache...")
        data = np.load(cache_file, allow_pickle=True)
        return data['features'], data['true_labels'].tolist()

    print("1. Đang tải dataset và trích xuất bằng CLIP + Kỹ thuật Zoom-in...")
    dataset = load_dataset("taesiri/imagenet-hard", split="validation")
    
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    features, true_labels = [], []
    
    for item in dataset:
        # Chuyển sang RGB
        img = item['image'].convert('RGB')
        
        # --- CHIẾN THUẬT ZOOM (CENTER CROP) ---
        # Theo README, việc zoom vào vùng đặc trưng giúp loại bỏ bối cảnh nhiễu
        # Ta lấy 75% diện tích vùng trung tâm để mô phỏng thao tác Zoom-in
        w, h = img.size
        zoom_factor = 0.75  # Giữ lại 75% vùng lõi
        left = (w - w * zoom_factor) / 2
        top = (h - h * zoom_factor) / 2
        right = (w + w * zoom_factor) / 2
        bottom = (h + h * zoom_factor) / 2
        img_zoomed = img.crop((left, top, right, bottom))
        
        # Đưa ảnh đã Zoom vào mô hình CLIP
        inputs = processor(images=img_zoomed, return_tensors="pt")
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
    # Lưu lại để các lần chạy sau không phải Zoom và trích xuất lại (tốn thời gian)
    np.savez(cache_file, features=features, true_labels=true_labels)
    return features, true_labels

# --- 2. MAIN (PCA + GMM CLUSTERING) ---
def main():
    # Bước 1: Lấy dữ liệu (Đã áp dụng kỹ thuật Zoom từ Paper)
    features, true_labels = load_or_extract_features()
    
    # Bước 2: Nén đặc trưng bằng PCA
    # Giảm chiều giúp GMM tính toán ma trận hiệp phương sai nhanh hơn và tránh nhiễu
    print("\n2. Đang nén đặc trưng bằng PCA (giảm xuống 256 chiều)...")
    pca = PCA(n_components=256, random_state=42)
    compressed_features = pca.fit_transform(features)
    
    # Bước 3: Phân cụm bằng GMM (Gaussian Mixture Model)
    # GMM linh hoạt hơn K-Means vì cho phép các cụm có hình dạng elip khác nhau
    k_components_list = [800, 1000, 1200]
    
    best_ari = -1
    best_labels = None
    best_k = 0

    print("\n--- BẮT ĐẦU CHẠY GMM TRÊN DỮ LIỆU ĐÃ TỐI ƯU ZOOM ---")
    for k_clusters in k_components_list:
        print(f"Đang chạy GMM với {k_clusters} cụm...")
        
        # covariance_type='diag' phù hợp cho dữ liệu lớn, đảm bảo tốc độ xử lý
        gmm = GaussianMixture(n_components=k_clusters, covariance_type='diag', random_state=42, max_iter=100)
        pred_labels = gmm.fit_predict(compressed_features)
        
        ari = adjusted_rand_score(true_labels, pred_labels)
        print(f"-> Thử K={k_clusters} cụm | Điểm ARI: {ari:.4f}")
        
        if ari > best_ari:
            best_ari = ari
            best_labels = pred_labels
            best_k = k_clusters

    print("\n--- KẾT QUẢ TỐI ƯU NHẤT ---")
    print(f"Số lượng cụm: {best_k} | ARI: {best_ari:.4f}")
    
    # Gom nhóm để đánh giá Purity
    clusters = {}
    for node_idx, cluster_id in enumerate(best_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node_idx)
        
    communities = list(clusters.values())
    sorted_communities = sorted(communities, key=len, reverse=True)
    
    print("\n--- TOP 5 CỤM LỚN NHẤT ĐỂ ĐÁNH GIÁ ĐỘ TINH KHIẾT (PURITY) ---")
    for i, comm in enumerate(sorted_communities[:5]):
        labels_in_cluster = [true_labels[node] for node in comm]
        top_label, top_count = Counter(labels_in_cluster).most_common(1)[0]
        purity = (top_count / len(comm)) * 100
        print(f"Cụm {i+1} ({len(comm)} ảnh) -> Nhãn đa số: {top_label} (Purity: {purity:.1f}%)")

    # --- 3. XUẤT ẢNH RA THƯ MỤC ĐỂ BÁO CÁO ---
    print("\n--- ĐANG XUẤT ẢNH RA THƯ MỤC 'Image_GMM_Zoomed' ---")
    dataset = load_dataset("taesiri/imagenet-hard", split="validation")
    main_output_dir = "Image_GMM_Zoomed"
    os.makedirs(main_output_dir, exist_ok=True)
    
    num_clusters_to_export = min(100, len(sorted_communities))
    for i, comm in enumerate(sorted_communities[:num_clusters_to_export]):
        cluster_dir = os.path.join(main_output_dir, f"Cluster_{i+1}")
        os.makedirs(cluster_dir, exist_ok=True)
        for img_id in comm:
            img = dataset[img_id]['image'].convert('RGB')
            img.save(os.path.join(cluster_dir, f"img_ID{img_id}.jpg"))
            
    print(f"-> HOÀN TẤT! Hãy kiểm tra thư mục {main_output_dir} để xem sự khác biệt.")

if __name__ == "__main__":
    main()