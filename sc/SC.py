import os
os.environ["HF_DATASETS_DISABLE_LOCKS"] = "1"

import time
from collections import Counter
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from datasets import load_dataset
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CẤU HÌNH CHUNG
# =============================================================================
KNN_METRIC = 'cosine'
ALPHA      = 0.5  # score = ALPHA * ARI + (1 - ALPHA) * NMI

# -----------------------------------------------------------------------------
# PHASE 1 — BASELINE
# -----------------------------------------------------------------------------
BASELINE_N_NEIGHBORS = 20
BASELINE_THRESHOLD   = 0.25
BASELINE_K           = 100

# -----------------------------------------------------------------------------
# PHASE 2 — TUNING
# -----------------------------------------------------------------------------
# Chỉ chạy với cấu hình tối ưu để tối đa hóa tốc độ chạy
TUNING_N_NEIGHBORS_LIST = [20, 30, 40]
TUNING_THRESHOLD_LIST   = [0.1, 0.2, 0.25, 0.3]
TUNING_K_LIST           = [200, 500, 700, 1000]


# =============================================================================
# HÀM TÍNH SCORE & PHỤ TRỢ
# =============================================================================
def compute_score(ari, nmi, alpha=ALPHA):
    return alpha * ari + (1 - alpha) * nmi


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


def run_spectral_clustering(similarity_matrix, k):
    try:
        sc = SpectralClustering(
            n_clusters=k,
            affinity='precomputed',
            assign_labels='cluster_qr',
            eigen_solver='amg',
            random_state=42
        )
        return sc.fit_predict(similarity_matrix)
    except Exception:
        sc = SpectralClustering(
            n_clusters=k,
            affinity='precomputed',
            assign_labels='cluster_qr',
            eigen_solver='arpack',
            random_state=42
        )
        return sc.fit_predict(similarity_matrix)


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


# =============================================================================
# PHASE 1: BASELINE
# =============================================================================
def run_baseline(features_data, true_labels):
    print("\n" + "="*80)
    print("PHASE 1: BASELINE")
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

    phase_elapsed = time.time() - t_start
    print(f"--> Tổng thời gian PHASE 1: {phase_elapsed:.2f}s")

    return {"n_neighbors": BASELINE_N_NEIGHBORS, "threshold": BASELINE_THRESHOLD,
            "k": BASELINE_K, "ari": ari, "nmi": nmi, "score": score,
            "labels": labels, "sim_matrix": sim_matrix, "time": phase_elapsed}


# =============================================================================
# PHASE 2: FULL TUNING
# =============================================================================
def run_tuning(features_data, true_labels):
    total_runs = len(TUNING_N_NEIGHBORS_LIST) * len(TUNING_THRESHOLD_LIST) * len(TUNING_K_LIST)

    print("\n" + "="*80)
    print("PHASE 2: TUNING")
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

    phase_elapsed = time.time() - t_start
    print(f"--> Tổng thời gian PHASE 2: {phase_elapsed:.2f}s")

    return best


# =============================================================================
# MAIN
# =============================================================================
def main():
    t_total_start = time.time()

    # 1. Nạp features
    try:
        features_baseline = np.load('features_no_umap.npy')
        print(f"--> Đã nạp đặc trưng (features_no_umap.npy): shape={features_baseline.shape}")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'features_no_umap.npy'")
        return

    try:
        features_tuned = np.load('features.npy')
        print(f"--> Đã nạp đặc trưng (features.npy): shape={features_tuned.shape}")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'features.npy'")
        return

    # 2. Ground Truth
    print("--> Đang tải Ground Truth...")
    labels_file = 'labels.npy'
    # Check current directory, parent directory or subfolders
    for path in [labels_file, os.path.join('..', labels_file), os.path.join('..', 'labels', labels_file)]:
        if os.path.exists(path):
            labels_file = path
            break
    
    true_labels = None
    if os.path.exists(labels_file):
        try:
            true_labels = np.load(labels_file).tolist()
            print(f"--> Đã nạp Ground Truth từ file cục bộ: {labels_file}")
        except Exception:
            pass

    if true_labels is None:
        try:
            dataset = load_dataset("taesiri/imagenet-hard", split="validation", local_files_only=True)
        except Exception:
            print("--> Không thể tải offline, đang thử tải online từ Hugging Face...")
            dataset = load_dataset("taesiri/imagenet-hard", split="validation")
        true_labels = [
            item['label'][0] if isinstance(item['label'], list) else item['label']
            for item in dataset
        ]
        # Save to local file labels.npy in parent directory so it is shared
        try:
            np.save(os.path.join('..', 'labels.npy'), np.array(true_labels))
            print("--> Đã lưu Ground Truth ra file cục bộ '../labels.npy' để tăng tốc cho các thuật toán khác.")
        except Exception:
            np.save('labels.npy', np.array(true_labels))
            print("--> Đã lưu Ground Truth ra file cục bộ 'labels.npy'.")

    if len(true_labels) != len(features_baseline) or len(true_labels) != len(features_tuned):
        print("Lỗi: Số nhãn không khớp với số features.")
        return

    # 3. Chạy 2 phase
    baseline_result = run_baseline(features_baseline, true_labels)
    tuned_result    = run_tuning(features_tuned, true_labels)

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
    time_str = f"{h} giờ {m} phút {s:.2f} giây" if h > 0 else (f"{m} phút {s:.2f} giây" if m > 0 else f"{s:.2f} giây")
    print(f"\n--> TỔNG THỜI GIAN CHẠY TOÀN BỘ CHƯƠNG TRÌNH: {time_str}")


if __name__ == "__main__":
    main()