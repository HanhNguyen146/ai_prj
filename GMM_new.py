import os
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from collections import Counter

# --- 1. TRÍCH XUẤT ĐA GÓC NHÌN (ENSEMBLE FEATURES) ---
# Đây chính là bước "Fine-tuning" về mặt dữ liệu để cải thiện độ chính xác
def extract_ensemble_features(cache_file="features_ensemble_fixed.npz"):
    if os.path.exists(cache_file):
        print("1. Đang tải đặc trưng Ensemble (Gốc + Zoom) từ file cache...")
        data = np.load(cache_file, allow_pickle=True)
        return data['features'], data['true_labels'].tolist()

    print("1. Đang trích xuất đặc trưng kết hợp (Gốc + Zoom) theo chiến thuật TTA...")
    dataset = load_dataset("taesiri/imagenet-hard", split="validation")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    features, true_labels = [], []
    for item in dataset:
        img = item['image'].convert('RGB')
        
        # Ảnh 1: Nguyên bản
        # Ảnh 2: Zoom vùng tâm (Center Crop 80%) theo gợi ý từ README
        w, h = img.size
        img_zoomed = img.crop((w*0.1, h*0.1, w*0.9, h*0.9))
        
        # Đưa cả 2 ảnh vào Processor
        inputs = processor(images=[img, img_zoomed], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            
            # --- SỬA LỖI TẠI ĐÂY ---
            # Kiểm tra nếu outputs là object thì lấy thuộc tính tensor ra
            if not isinstance(outputs, torch.Tensor):
                outputs = getattr(outputs, "pooler_output", outputs)
            
            # Tính trung bình cộng của 2 vector (Ảnh Gốc và Ảnh Zoom)
            # outputs có shape (2, 512), dim=0 sẽ gộp 2 hàng thành 1
            ensemble_vec = outputs.mean(dim=0).cpu().numpy()
        
        features.append(ensemble_vec)
        label = item['label']
        true_labels.append(label[0] if isinstance(label, list) else label)

    features = np.array(features)
    np.savez(cache_file, features=features, true_labels=true_labels)
    return features, true_labels

# --- 2. MAIN (THỰC HIỆN PHÂN CỤM GMM TỐI ƯU) ---
def main():
    # Bước 1: Lấy dữ liệu đã được Ensemble (Fine-tuned)
    feat, labels = extract_ensemble_features()
    
    # Bước 2: Chuẩn hóa bằng PCA giữ lại 95% thông tin
    # Đây là bước "Chuẩn hóa hệ thống" giúp giảm nhiễu nhưng vẫn giữ đủ đặc trưng
    print("\n2. Đang chuẩn hóa và nén đặc trưng bằng PCA (giữ 95% phương sai)...")
    pca = PCA(n_components=0.95, random_state=42)
    feat_pca = pca.fit_transform(feat)
    print(f"-> Hệ thống giữ lại: {feat_pca.shape[1]} chiều đặc trưng quan trọng.")

    # Bước 3: Chạy GMM (Thuật toán đại diện cho phân cụm xác suất)
    print("\n3. Đang thực hiện phân cụm GMM trên không gian đặc trưng mới...")
    gmm = GaussianMixture(n_components=1000, covariance_type='diag', random_state=42)
    pred_labels = gmm.fit_predict(feat_pca)
    
    # Bước 4: Đánh giá bằng ARI
    ari = adjusted_rand_score(labels, pred_labels)
    print(f"\n--- KẾT QUẢ GMM SAU KHI FINE-TUNING (ENSEMBLE) ---")
    print(f"Điểm ARI đạt được: {ari:.4f}")

    # Gom nhóm kết quả để hiển thị Top cụm như yêu cầu trong ảnh đề tài
    clusters = {}
    for node_idx, cluster_id in enumerate(pred_labels):
        if cluster_id not in clusters: clusters[cluster_id] = []
        clusters[cluster_id].append(node_idx)
    
    sorted_comm = sorted(clusters.values(), key=len, reverse=True)
    print("\n--- TOP 3 CỤM LỚN NHẤT ---")
    for i, comm in enumerate(sorted_comm[:3]):
        labels_in_cluster = [labels[node] for node in comm]
        top_label, top_count = Counter(labels_in_cluster).most_common(1)[0]
        purity = (top_count / len(comm)) * 100
        print(f"Cụm {i+1} ({len(comm)} ảnh): Nhãn đa số {top_label} (Purity: {purity:.1f}%)")

if __name__ == "__main__":
    main()