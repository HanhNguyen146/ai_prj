import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import umap
from sklearn.neighbors import NearestNeighbors
import os

# =========================================================
# CẤU HÌNH
# =========================================================
FEATURES_PATH = "features_no_umap.npy"
BASELINE_LABELS = "louvain_baseline_labels.npy"
TUNED_LABELS = "louvain_tuned_labels.npy"

# =========================================================
# HÀM VẼ ẢNH
# =========================================================
def draw_visualization(coords_2d, features, labels, title_main, subtitle1, subtitle2, filename, k_neighbors=20, threshold=0.25):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor('white')
    
    # Tiêu đề tổng
    fig.suptitle(title_main, fontsize=28, color="#002060", y=0.95, fontweight='bold')

    # ---------------------------------------------------------
    # HÌNH 1: SCATTER PLOT (Toàn cảnh 10980 ảnh)
    # ---------------------------------------------------------
    ax1.set_title(subtitle1, fontsize=12, fontweight='bold', pad=15)
    scatter = ax1.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap='tab20', s=3, alpha=0.7)
    ax1.axis('off')

    # ---------------------------------------------------------
    # HÌNH 2: k-NN GRAPH (Lấy mẫu 300 nodes)
    # ---------------------------------------------------------
    ax2.set_title(subtitle2, fontsize=12, fontweight='bold', pad=15)
    
    # Lấy mẫu ngẫu nhiên 300 điểm
    np.random.seed(42)
    sample_indices = np.random.choice(len(features), size=300, replace=False)
    sample_feat = features[sample_indices]
    sample_coords = coords_2d[sample_indices]
    sample_labels = labels[sample_indices]

    # Xây dựng đồ thị k-NN cho 300 điểm này
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine').fit(sample_feat)
    distances, indices = nbrs.kneighbors(sample_feat)
    
    G = nx.Graph()
    for i in range(len(sample_indices)):
        G.add_node(i, pos=(sample_coords[i, 0], sample_coords[i, 1]))
        for j_idx in range(1, k_neighbors + 1):
            j = indices[i, j_idx]
            sim = 1.0 - distances[i, j_idx]
            if sim > threshold:
                G.add_edge(i, j)

    # Vẽ đồ thị
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.2, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=sample_labels, cmap='tab20', node_size=30, alpha=0.9)
    ax2.axis('off')

    # Lưu ảnh
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Đã lưu ảnh: {filename}")
    plt.close()

# =========================================================
# CHẠY CHÍNH
# =========================================================
def main():
    print("1. Đang nạp features và labels...")
    features = np.load(FEATURES_PATH)
    labels_base = np.load(BASELINE_LABELS)
    labels_tuned = np.load(TUNED_LABELS)

    print("2. Đang chạy UMAP để giảm xuống 2D (Sẽ mất khoảng 1-2 phút)...")
    # Nén 768 chiều xuống 2 chiều để vẽ đồ thị
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    coords_2d = reducer.fit_transform(features)

    print("3. Đang vẽ ảnh Baseline...")
    draw_visualization(
        coords_2d, features, labels_base,
        title_main="TRỰC QUAN HÓA KẾT QUẢ BASELINE",
        subtitle1="Toàn cảnh 10980 ảnh\nLOUVAIN BASELINE | n_nbrs=20 | thr=0.30",
        subtitle2="k-NN Graph (mẫu 300 nodes)\nLOUVAIN BASELINE | n_nbrs=20 | thr=0.30",
        filename="Louvain_Baseline_Visualization.png",
        k_neighbors=20, threshold=0.30
    )

    print("4. Đang vẽ ảnh Tuned...")
    draw_visualization(
        coords_2d, features, labels_tuned,
        title_main="TRỰC QUAN HÓA KẾT QUẢ TUNED",
        subtitle1="Toàn cảnh 10980 ảnh\nLOUVAIN TUNED | Tối ưu hóa tham số",
        subtitle2="k-NN Graph (mẫu 300 nodes)\nLOUVAIN TUNED | Tối ưu hóa tham số",
        filename="Louvain_Tuned_Visualization.png",
        k_neighbors=20, threshold=0.20 # Sửa threshold ở đây thành threshold của cấu hình BEST của bạn
    )
    print("HOÀN THÀNH!")

if __name__ == "__main__":
    main()