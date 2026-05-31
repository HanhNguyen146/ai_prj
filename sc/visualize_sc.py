import os
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

KNN_METRIC            = 'cosine'
N_NODES_VISUALIZATION = 300

def build_similarity_matrix(features, n_neighbors, threshold):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=KNN_METRIC, n_jobs=-1)
    nbrs.fit(features)
    distance_matrix   = nbrs.kneighbors_graph(mode='distance')
    similarity_matrix = distance_matrix.copy()
    similarity_matrix.data = 1.0 - similarity_matrix.data
    similarity_matrix.data[similarity_matrix.data <= threshold] = 0
    similarity_matrix.eliminate_zeros()
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    similarity_matrix.eliminate_zeros()
    return similarity_matrix

def plot_visualization(features, labels, n_clusters, adjacency_matrix,
                       title_suffix, filename,
                       num_nodes_subset=N_NODES_VISUALIZATION):
    print(f"--> Đang chạy t-SNE cho '{filename}'...")
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42)
    x_tsne = tsne.fit_transform(features)

    print(f"--> Đang vẽ đồ thị...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    cmap = plt.get_cmap('tab20')
    norm = colors.Normalize(vmin=0, vmax=n_clusters - 1)

    ax1.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, cmap=cmap, norm=norm, s=3, alpha=0.5)
    ax1.set_title(f"Toàn cảnh {features.shape[0]} ảnh — {n_clusters} cụm\n{title_suffix}",
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    graph_knn    = nx.from_scipy_sparse_array(adjacency_matrix)
    rng          = np.random.default_rng(123)
    n_sample     = min(num_nodes_subset, graph_knn.number_of_nodes())
    random_nodes = rng.choice(list(graph_knn.nodes()), n_sample, replace=False)
    g_sub        = graph_knn.subgraph(random_nodes)
    pos          = {n: x_tsne[n] for n in random_nodes}
    node_colors  = [cmap(norm(labels[n])) for n in random_nodes]

    nx.draw_networkx(g_sub, pos=pos, ax=ax2, with_labels=False,
                     node_size=60, node_color=node_colors,
                     edge_color='gray', width=0.4, alpha=0.8)
    ax2.set_title(f"k-NN Graph (mẫu {n_sample} nodes)\n{title_suffix}",
                  fontsize=14, fontweight='bold')
    ax2.axis('off')

    fig.suptitle("Spectral Clustering — " + title_suffix, fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Đã lưu đồ thị trực quan hóa vào '{filename}'.")

def main():
    print("================================================================================")
    print("VẼ ĐỒ THỊ TRỰC QUAN HÓA SPECTRAL CLUSTERING")
    print("================================================================================")

    # Đảm bảo thư mục image tồn tại
    os.makedirs("image", exist_ok=True)

    # 1. Vẽ Baseline
    try:
        features_baseline = np.load('features_no_umap.npy')
        labels_baseline = np.load('spectral_baseline_labels.npy')
        print("--> Đã nạp features và nhãn Baseline.")
        
        sim_matrix = build_similarity_matrix(features_baseline, 20, 0.25)
        plot_visualization(
            features_baseline, labels_baseline, 100, sim_matrix,
            title_suffix="BASELINE | n_nbrs=20 | thr=0.25 | K=100",
            filename="image/SC_Baseline_Visualization.png"
        )
    except FileNotFoundError as e:
        print(f"--> Không thể vẽ Baseline: thiếu file {e.filename}")

    # 2. Vẽ Tuned
    try:
        features_tuned = np.load('features.npy')
        labels_tuned = np.load('spectral_tuned_labels.npy')
        print("--> Đã nạp features và nhãn Tuned.")
        
        sim_matrix = build_similarity_matrix(features_tuned, 20, 0.2)
        plot_visualization(
            features_tuned, labels_tuned, 1000, sim_matrix,
            title_suffix="TUNED BEST | n_nbrs=20 | thr=0.2 | K=1000",
            filename="image/SC_Tuned_Visualization.png"
        )
    except FileNotFoundError as e:
        print(f"--> Không thể vẽ Tuned: thiếu file {e.filename}")

if __name__ == "__main__":
    main()
