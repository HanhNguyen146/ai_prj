import os
import shutil
import time
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from datasets import load_dataset
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CẤU HÌNH CHUNG
# =============================================================================
KNN_METRIC            = 'cosine'
N_NODES_VISUALIZATION = 300

# -----------------------------------------------------------------------------
# THAM SỐ ĐIỀU HÒA ALPHA
# -----------------------------------------------------------------------------
# score = ALPHA * ARI + (1 - ALPHA) * NMI
#
# ALPHA = 1.0 → chỉ quan tâm ARI
# ALPHA = 0.0 → chỉ quan tâm NMI
# ALPHA = 0.5 → cân bằng ARI và NMI
# ALPHA = 0.6 → nghiêng về ARI nhưng vẫn xét NMI
#
# Gợi ý: dùng 0.5 cho kết quả cân bằng nhất
ALPHA = 0.5

# -----------------------------------------------------------------------------
# PHASE 1 — BASELINE
# -----------------------------------------------------------------------------
BASELINE_N_NEIGHBORS = 20
BASELINE_THRESHOLD   = 0.25
BASELINE_K           = 100
BASELINE_VIZ_FILE    = "SC_Baseline_Visualization.png"
BASELINE_FOLDER      = "Cluster_Results_SC_Baseline"

# -----------------------------------------------------------------------------
# PHASE 2 — FULL TUNING
# -----------------------------------------------------------------------------
TUNING_N_NEIGHBORS_LIST = [20, 30, 40]
TUNING_THRESHOLD_LIST   = [0.1, 0.2, 0.25, 0.3]
TUNING_K_LIST           = [200, 500, 700, 1000]
TUNING_VIZ_FILE         = "SC_Tuned_Visualization.png"
TUNING_FOLDER           = "Cluster_Results_SC_Tuned"


# =============================================================================
# HÀM TÍNH SCORE
# =============================================================================
def compute_score(ari, nmi, alpha=ALPHA):
    """
    Tính điểm tổng hợp từ ARI và NMI theo tham số alpha.
    score = alpha * ARI + (1 - alpha) * NMI
    """
    return alpha * ari + (1 - alpha) * nmi


# =============================================================================
# HÀM PHỤ TRỢ DÙNG CHUNG
# =============================================================================
def build_similarity_matrix(features, n_neighbors, threshold):
    """Xây KNN graph → similarity → symmetrize → threshold."""
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


def run_spectral_clustering(similarity_matrix, k):
    sc = SpectralClustering(
        n_clusters=k,
        affinity='precomputed',
        assign_labels='cluster_qr',
        random_state=42
    )
    return sc.fit_predict(similarity_matrix)


def _save_cluster_images(img_indices, cluster_dir, dataset):
    for img_idx in img_indices:
        try:
            img = dataset[img_idx]['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(os.path.join(cluster_dir, f"img_ID{img_idx}.jpg"))
        except Exception as e:
            print(f"  Lỗi khi lưu ảnh ID {img_idx}: {e}")


def export_clusters_to_folders(labels, dataset, output_folder, max_images_per_cluster=None):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    clusters = {}
    for node_idx, cluster_id in enumerate(labels):
        clusters.setdefault(int(cluster_id), []).append(node_idx)

    for cluster_id, img_indices in sorted(clusters.items(), key=lambda x: -len(x[1])):
        folder_name = f"Cluster_{cluster_id + 1:04d}_({len(img_indices)}anh)"
        cluster_dir = os.path.join(output_folder, folder_name)
        os.makedirs(cluster_dir)
        sample = img_indices[:max_images_per_cluster] if max_images_per_cluster else img_indices
        _save_cluster_images(sample, cluster_dir, dataset)

    print(f"--> Đã lưu {len(clusters)} cụm ảnh vào '{output_folder}'.")


def print_top5_clusters(labels, true_labels):
    clusters = {}
    for node_idx, cluster_id in enumerate(labels):
        clusters.setdefault(int(cluster_id), []).append(node_idx)

    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
    print("--- CHI TIẾT TOP 5 CỤM LỚN NHẤT ---")
    for i, comm in enumerate(sorted_clusters[:5]):
        lbl_in_cluster = [true_labels[n] for n in comm]
        top_label, top_count = Counter(lbl_in_cluster).most_common(1)[0]
        purity = (top_count / len(comm)) * 100
        print(f"Cụm {i+1} ({len(comm)} ảnh) -> Nhãn gốc chiếm đa số: {top_label} (Purity: {purity:.1f}%) | ID đại diện: {list(comm)[:5]}")
    print("-" * 80)


def plot_visualization(features, labels, n_clusters, adjacency_matrix,
                       title_suffix, filename,
                       num_nodes_subset=N_NODES_VISUALIZATION):
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42)
    x_tsne = tsne.fit_transform(features)

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
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Đã lưu đồ thị trực quan hóa vào '{filename}'.")


# =============================================================================
# PHASE 1: BASELINE
# =============================================================================
def run_baseline(features_data, true_labels, dataset):
    print("\n" + "="*80)
    print("PHASE 1: BASELINE (features.npy)")
    print("="*80)

    t_start = time.time()

    t_graph    = time.time()
    sim_matrix = build_similarity_matrix(features_data, BASELINE_N_NEIGHBORS, BASELINE_THRESHOLD)
    t_graph_elapsed = time.time() - t_graph

    t0      = time.time()
    labels  = run_spectral_clustering(sim_matrix, BASELINE_K)
    t_sc_elapsed = time.time() - t0

    ari   = adjusted_rand_score(true_labels, labels)
    nmi   = normalized_mutual_info_score(true_labels, labels)
    score = compute_score(ari, nmi)

    print("\n[BASELINE RESULT]")
    print(f"  Params : n_neighbors={BASELINE_N_NEIGHBORS}, threshold={BASELINE_THRESHOLD}, n_clusters={BASELINE_K}")
    print(f"  Graph  : {sim_matrix.nnz} cạnh | Thời gian xây đồ thị: {t_graph_elapsed:.2f}s")
    print(f"  Clustering Time: {t_sc_elapsed:.2f}s")
    print(f"  Metrics: ARI={ari:.4f} | NMI={nmi:.4f} | Score={score:.4f}")
    print("="*80)

    print_top5_clusters(labels, true_labels)

    np.save('spectral_baseline_labels.npy', labels)

    t_io = time.time()
    export_clusters_to_folders(labels, dataset, BASELINE_FOLDER, max_images_per_cluster=None)
    t_io_elapsed = time.time() - t_io

    t_viz = time.time()
    plot_visualization(
        features_data, labels, BASELINE_K, sim_matrix,
        title_suffix=f"BASELINE | n_nbrs={BASELINE_N_NEIGHBORS} | thr={BASELINE_THRESHOLD} | K={BASELINE_K}",
        filename=BASELINE_VIZ_FILE
    )
    t_viz_elapsed = time.time() - t_viz

    phase_elapsed = time.time() - t_start
    print(f"--> Tổng thời gian PHASE 1: {phase_elapsed:.2f}s (Xuất ảnh: {t_io_elapsed:.2f}s, Trực quan hóa: {t_viz_elapsed:.2f}s)")

    return {"n_neighbors": BASELINE_N_NEIGHBORS, "threshold": BASELINE_THRESHOLD,
            "k": BASELINE_K, "ari": ari, "nmi": nmi, "score": score,
            "labels": labels, "sim_matrix": sim_matrix, "time": phase_elapsed}


# =============================================================================
# PHASE 2: FULL TUNING
# =============================================================================
def run_tuning(features_data, true_labels, dataset):
    total_runs = len(TUNING_N_NEIGHBORS_LIST) * len(TUNING_THRESHOLD_LIST) * len(TUNING_K_LIST)

    print("\n" + "="*80)
    print("PHASE 2: FULL TUNING (features_no_umap.npy)")
    print(f"  Tổng số cấu hình chạy thử: {total_runs}")
    print("="*80)

    t_start = time.time()

    all_results = []
    best        = {"score": -1}
    run_count   = 0

    for n_nbrs in TUNING_N_NEIGHBORS_LIST:
        for thr in TUNING_THRESHOLD_LIST:
            t_graph    = time.time()
            sim_matrix = build_similarity_matrix(features_data, n_nbrs, thr)
            t_graph_elapsed = time.time() - t_graph

            for k in TUNING_K_LIST:
                run_count += 1
                print(f"  [{run_count:>2}/{total_runs}] n_nbrs={n_nbrs}, thr={thr}, K={k}...", end=" ", flush=True)

                t0      = time.time()
                labels  = run_spectral_clustering(sim_matrix, k)
                elapsed = time.time() - t0
                ari     = adjusted_rand_score(true_labels, labels)
                nmi     = normalized_mutual_info_score(true_labels, labels)
                score   = compute_score(ari, nmi)

                result = {
                    "n_neighbors": n_nbrs, "threshold": thr, "k": k,
                    "ari": ari, "nmi": nmi, "score": score,
                    "time": elapsed, "n_edges": sim_matrix.nnz,
                    "labels": labels, "sim_matrix": sim_matrix
                }
                all_results.append(result)
                print(f"ARI={ari:.4f} | NMI={nmi:.4f} | Score={score:.4f} ({elapsed:.1f}s)")

                if score > best["score"]:
                    best = result

    # Bảng tổng hợp: Chỉ hiện TOP 5 cấu hình tốt nhất để tinh gọn
    print("\n" + "="*90)
    print("TOP 5 CẤU HÌNH TỐT NHẤT")
    print(f"{'n_nbrs':>7} | {'thr':>5} | {'K':>5} | "
          f"{'ARI':>8} | {'NMI':>8} | {'Score':>8} | {'Time':>7} | {'edges':>8}")
    print("-"*80)
    for r in sorted(all_results, key=lambda x: -x['score'])[:5]:
        is_best = (r['n_neighbors'] == best['n_neighbors'] and
                   r['threshold']   == best['threshold'] and
                   r['k']           == best['k'])
        marker = " ◄ BEST" if is_best else ""
        print(f"{r['n_neighbors']:>7} | {r['threshold']:>5} | {r['k']:>5} | "
              f"{r['ari']:>8.4f} | {r['nmi']:>8.4f} | {r['score']:>8.4f} | "
              f"{r['time']:>6.1f}s | {r['n_edges']:>8}{marker}")
    print("="*90)

    print(f"\n[BEST TUNED CONFIGURATION]")
    print(f"  Params : n_neighbors={best['n_neighbors']}, threshold={best['threshold']}, n_clusters={best['k']}")
    print(f"  Graph  : {best['n_edges']} cạnh")
    print(f"  Metrics: ARI={best['ari']:.4f} | NMI={best['nmi']:.4f} | Score={best['score']:.4f}")
    print("="*90)

    print_top5_clusters(best['labels'], true_labels)

    np.save('spectral_tuned_labels.npy', best['labels'])

    t_io = time.time()
    export_clusters_to_folders(best['labels'], dataset, TUNING_FOLDER, max_images_per_cluster=None)
    t_io_elapsed = time.time() - t_io

    t_viz = time.time()
    plot_visualization(
        features_data, best['labels'], best['k'], best['sim_matrix'],
        title_suffix=(f"TUNED BEST (α={ALPHA:.1f}) | n_nbrs={best['n_neighbors']} | "
                      f"thr={best['threshold']} | K={best['k']}"),
        filename=TUNING_VIZ_FILE
    )
    t_viz_elapsed = time.time() - t_viz

    phase_elapsed = time.time() - t_start
    print(f"--> Tổng thời gian PHASE 2: {phase_elapsed:.2f}s (Xuất ảnh: {t_io_elapsed:.2f}s, Trực quan hóa: {t_viz_elapsed:.2f}s)")

    return best


# =============================================================================
# MAIN
# =============================================================================
def main():
    t_total_start = time.time()

    # 1. Nạp features
    try:
        features_baseline = np.load('features.npy')
        print(f"--> Đã nạp features baseline (features.npy): shape={features_baseline.shape}, dtype={features_baseline.dtype}")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'features.npy'")
        return

    try:
        features_tuned = np.load('features_no_umap.npy')
        print(f"--> Đã nạp features tuned (features_no_umap.npy): shape={features_tuned.shape}, dtype={features_tuned.dtype}")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'features_no_umap.npy'")
        return

    # 2. Ground Truth
    print("--> Đang tải dataset từ Hugging Face để lấy Ground Truth...")
    dataset = load_dataset("taesiri/imagenet-hard", split="validation")
    true_labels = [
        item['label'][0] if isinstance(item['label'], list) else item['label']
        for item in dataset
    ]
    if len(true_labels) != len(features_baseline) or len(true_labels) != len(features_tuned):
        print("Lỗi: Số nhãn không khớp với số features.")
        return

    # 3. Chạy 2 phase
    baseline_result = run_baseline(features_baseline, true_labels, dataset)
    tuned_result    = run_tuning(features_tuned, true_labels, dataset)

    # 4. So sánh cuối
    print("\n" + "="*80)
    print(f"SO SÁNH BASELINE vs BEST TUNED  (Score = {ALPHA:.1f}×ARI + {1-ALPHA:.1f}×NMI)")
    print(f"{'':25} {'BASELINE':>12} {'BEST TUNED':>12} {'DELTA':>10}")
    print("-"*65)
    print(f"  {'n_neighbors':<22} {baseline_result['n_neighbors']:>12} {tuned_result['n_neighbors']:>12}")
    print(f"  {'threshold':<22} {baseline_result['threshold']:>12} {tuned_result['threshold']:>12}")
    print(f"  {'n_clusters':<22} {baseline_result['k']:>12} {tuned_result['k']:>12}")
    print(f"  {'ARI':<22} {baseline_result['ari']:>12.4f} {tuned_result['ari']:>12.4f} "
          f"{tuned_result['ari'] - baseline_result['ari']:>+10.4f}")
    print(f"  {'NMI':<22} {baseline_result['nmi']:>12.4f} {tuned_result['nmi']:>12.4f} "
          f"{tuned_result['nmi'] - baseline_result['nmi']:>+10.4f}")
    print(f"  {'Score':<22} {baseline_result['score']:>12.4f} {tuned_result['score']:>12.4f} "
          f"{tuned_result['score'] - baseline_result['score']:>+10.4f}")
    print("="*80)

    total_elapsed = time.time() - t_total_start
    h = int(total_elapsed // 3600)
    m = int((total_elapsed % 3600) // 60)
    s = total_elapsed % 60
    if h > 0:
        time_str = f"{h} giờ {m} phút {s:.2f} giây"
    elif m > 0:
        time_str = f"{m} phút {s:.2f} giây"
    else:
        time_str = f"{s:.2f} giây"
    print(f"\n--> TỔNG THỜI GIAN CHẠY TOÀN BỘ CHƯƠNG TRÌNH: {time_str}")


if __name__ == "__main__":
    main()