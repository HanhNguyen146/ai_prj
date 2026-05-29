import os
import shutil
import time
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CẤU HÌNH CHUNG
# =============================================================================
PRECOMPUTED_FEATURES_PATH = "features_no_umap.npy"   # (10980, 768), float32, L2-normalized
KNN_METRIC                = 'cosine'

# -----------------------------------------------------------------------------
# THAM SỐ ĐIỀU HÒA ALPHA
# -----------------------------------------------------------------------------
# score = ALPHA * ARI + (1 - ALPHA) * NMI
#
# ALPHA = 1.0 → chỉ quan tâm ARI
# ALPHA = 0.0 → chỉ quan tâm NMI
# ALPHA = 0.5 → cân bằng ARI và NMI
#
# Gợi ý: dùng 0.5 cho kết quả cân bằng nhất
ALPHA = 0.5

# -----------------------------------------------------------------------------
# PCA WHITENING  (dùng chung cho cả 2 phase)
# -----------------------------------------------------------------------------
USE_PCA    = True
PCA_DIM    = 256
PCA_WHITEN = True

# -----------------------------------------------------------------------------
# LOUVAIN  (dùng chung cho cả 2 phase)
# -----------------------------------------------------------------------------
AUTO_TUNE_RESOLUTION = True    # tự tìm resolution để n_clusters ≈ n_gt
RESOLUTION_TOLERANCE = 0.10    # chấp nhận sai số ±10% so với n_gt_classes
N_SEEDS              = 5
USE_CONSENSUS        = False

# -----------------------------------------------------------------------------
# POST-PROCESSING  (dùng chung cho cả 2 phase)
# -----------------------------------------------------------------------------
MIN_CLUSTER_SIZE = 5

# -----------------------------------------------------------------------------
# PHASE 1 — BASELINE
# -----------------------------------------------------------------------------
BASELINE_K_NEIGHBORS = 20
BASELINE_THRESHOLD   = 0.30
BASELINE_MUTUAL_KNN  = False
BASELINE_FOLDER      = "Cluster_Results_Louvain_Baseline"
BASELINE_VIZ_FILE    = "Louvain_Baseline_Distribution.png"

# -----------------------------------------------------------------------------
# PHASE 2 — FULL TUNING
# -----------------------------------------------------------------------------
TUNING_K_NEIGHBORS_LIST = [20, 25, 30, 40]
TUNING_THRESHOLD_LIST   = [0.15, 0.20, 0.25, 0.30]
TUNING_MUTUAL_KNN_LIST  = [False]          # thêm True nếu muốn thử mutual KNN
TUNING_FOLDER           = "Cluster_Results_Louvain_Tuned"
TUNING_VIZ_FILE         = "Louvain_Tuned_Distribution.png"

# -----------------------------------------------------------------------------
# OUTPUT & VISUALIZATION
# -----------------------------------------------------------------------------
OUTPUT_DIR           = "louvain_results_opt"
IMGS_PER_CLUSTER     = 16
GRID_COLS            = 4
THUMB_SIZE           = 128
MAX_CLUSTER_IMGS     = 100
SAVE_INDIVIDUAL_IMGS = True


# =============================================================================
# HÀM TÍNH SCORE
# =============================================================================
def compute_score(ari, nmi, alpha=ALPHA):
    """score = alpha * ARI + (1 - alpha) * NMI"""
    return alpha * ari + (1 - alpha) * nmi


# =============================================================================
# HÀM PHỤ TRỢ DÙNG CHUNG
# =============================================================================
def load_features_and_dataset():
    """Nạp features và dataset, trả về tuple đầy đủ."""
    print(f"--> Đang nạp features từ {PRECOMPUTED_FEATURES_PATH}…")
    t0       = time.time()
    features = np.load(PRECOMPUTED_FEATURES_PATH)
    print(f"--> Shape: {features.shape}  |  dtype: {features.dtype}  |  Time: {time.time()-t0:.2f}s")

    print("--> Đang tải dataset từ Hugging Face để lấy Ground Truth…")
    ds     = load_dataset("taesiri/imagenet-hard", split="validation")
    N      = len(ds)
    gt_raw = ds["label"]
    if isinstance(gt_raw[0], list):
        gt_raw = [l[0] for l in gt_raw]
    gt_raw = np.array(gt_raw)

    le           = LabelEncoder()
    gt_labels    = le.fit_transform(gt_raw)
    n_gt_classes = len(np.unique(gt_labels))

    english_labels = ds["english_label"]
    if isinstance(english_labels[0], list):
        english_labels = [l[0] for l in english_labels]

    print(f"--> Dataset: {N} ảnh  |  Ground-truth classes: {n_gt_classes}")
    return features, gt_raw, gt_labels, english_labels, ds, N, n_gt_classes


def preprocess_features(features):
    """L2 normalize → (optional) PCA whitening → re-normalize."""
    norms = np.linalg.norm(features, axis=1)
    if np.allclose(norms, 1.0, atol=1e-4):
        print("--> Features đã L2-normalized, bỏ qua bước normalize.")
        feat = features.copy()
    else:
        print(f"--> Norms: min={norms.min():.4f}, max={norms.max():.4f}. Đang normalize…")
        feat = features / (norms[:, None] + 1e-10)
    feat = feat.astype(np.float32)

    if USE_PCA:
        n_comp = PCA_DIM if PCA_DIM else feat.shape[1]
        print(f"--> PCA Whitening: {feat.shape[1]}d → {n_comp}d …")
        t_pca  = time.time()
        pca    = PCA(n_components=n_comp, whiten=PCA_WHITEN, random_state=42)
        feat   = pca.fit_transform(feat).astype(np.float32)
        norms2 = np.linalg.norm(feat, axis=1, keepdims=True)
        feat   = feat / (norms2 + 1e-10)
        print(f"--> Explained variance: {pca.explained_variance_ratio_.sum():.3f}  |  "
              f"Time: {time.time()-t_pca:.1f}s")
    return feat


def build_knn_graph(feat, N, k_neighbors, threshold, mutual_knn):
    """Xây dựng KNN graph → lọc theo threshold (và mutual nếu bật)."""
    t_graph = time.time()
    nbrs    = NearestNeighbors(n_neighbors=k_neighbors + 1, metric=KNN_METRIC,
                               n_jobs=-1, algorithm="brute")
    nbrs.fit(feat)
    distances, indices = nbrs.kneighbors(feat)
    distances    = distances[:, 1:]
    indices      = indices[:, 1:]
    similarities = 1.0 - distances

    neighbor_sets = [set(indices[i]) for i in range(N)]
    edges = {}
    for i in range(N):
        for j_idx in range(k_neighbors):
            j   = int(indices[i, j_idx])
            sim = float(similarities[i, j_idx])
            if sim <= threshold or i == j:
                continue
            if mutual_knn and i not in neighbor_sets[j]:
                continue
            key        = (min(i, j), max(i, j))
            edges[key] = max(edges.get(key, sim), sim)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from([(u, v, {"weight": w}) for (u, v), w in edges.items()])
    t_graph_elapsed = time.time() - t_graph
    print(f"--> {'Mutual ' if mutual_knn else ''}KNN Graph: "
          f"{G.number_of_edges():,} cạnh  |  Build time: {t_graph_elapsed:.2f}s")
    return G, t_graph_elapsed


def _run_louvain_once(G, resolution, seed):
    return community_louvain.best_partition(
        G, weight="weight", resolution=resolution, random_state=seed
    )


def auto_tune_resolution(G, n_gt_classes):
    """Binary search để tìm resolution sao cho n_clusters ≈ n_gt_classes ± tolerance."""
    target_lo = int(n_gt_classes * (1 - RESOLUTION_TOLERANCE))
    target_hi = int(n_gt_classes * (1 + RESOLUTION_TOLERANCE))
    lo, hi    = 0.1, 50.0
    best_res  = 1.0

    for hi_try in [50.0, 200.0, 500.0, 1000.0]:
        test_part = _run_louvain_once(G, hi_try, seed=42)
        n_test    = len(set(test_part.values()))
        print(f"  [warm-up] res={hi_try} → {n_test} clusters (cần ≥{target_lo})")
        if n_test >= target_lo:
            hi = hi_try
            break
    else:
        print(f"  [WARN] Max clusters: {n_test}. Dùng resolution={hi_try}.")
        return float(hi_try)

    for _ in range(25):
        mid      = (lo + hi) / 2.0
        part_mid = _run_louvain_once(G, mid, seed=42)
        n_mid    = len(set(part_mid.values()))
        print(f"    res={mid:.3f} → {n_mid} clusters (target {target_lo}–{target_hi})")
        if target_lo <= n_mid <= target_hi:
            best_res = mid
            print(f"--> Hội tụ! Resolution={best_res:.4f}, clusters={n_mid}")
            break
        lo = mid if n_mid < target_lo else lo
        hi = mid if n_mid > target_hi else hi

    return best_res


def run_multi_seed_louvain(G, resolution, N):
    """Chạy Louvain N_SEEDS lần → consensus hoặc chọn modularity cao nhất."""
    t2             = time.time()
    all_partitions = []
    for seed in range(N_SEEDS):
        p   = _run_louvain_once(G, resolution, seed=seed * 7 + 42)
        mod = community_louvain.modularity(p, G, weight="weight")
        print(f"  seed={seed*7+42:3d}  →  {len(set(p.values()))} clusters  |  modularity={mod:.4f}")
        all_partitions.append(p)

    if USE_CONSENSUS:
        co_assoc = np.zeros((N, N), dtype=np.float32)
        for p in all_partitions:
            labels_p  = np.array([p[i] for i in range(N)])
            co_assoc += (labels_p[:, None] == labels_p[None, :]).astype(np.float32)
        co_assoc /= N_SEEDS
        G_cons = nx.Graph()
        G_cons.add_nodes_from(range(N))
        rows, cols_idx = np.where(co_assoc > 0.5)
        for r, c in zip(rows, cols_idx):
            if r < c:
                G_cons.add_edge(r, c, weight=float(co_assoc[r, c]))
        final_partition = _run_louvain_once(G_cons, resolution, seed=42)
        print(f"--> Consensus graph edges: {G_cons.number_of_edges():,}")
    else:
        best_mod, final_partition = -1.0, None
        for p in all_partitions:
            mod = community_louvain.modularity(p, G, weight="weight")
            if mod > best_mod:
                best_mod, final_partition = mod, p
        print(f"--> Chọn partition tốt nhất: modularity={best_mod:.4f}")

    louvain_labels = np.array([final_partition[i] for i in range(N)])
    return louvain_labels, final_partition, time.time() - t2


def merge_small_clusters(louvain_labels, feat):
    """Merge cluster < MIN_CLUSTER_SIZE vào nearest centroid."""
    if MIN_CLUSTER_SIZE <= 1:
        return louvain_labels
    unique_cls     = np.unique(louvain_labels)
    centroids      = {c: feat[louvain_labels == c].mean(axis=0) for c in unique_cls}
    small_cls      = [c for c in unique_cls if (louvain_labels == c).sum() < MIN_CLUSTER_SIZE]
    large_cls      = [c for c in unique_cls if (louvain_labels == c).sum() >= MIN_CLUSTER_SIZE]
    if small_cls and large_cls:
        large_centroids = np.array([centroids[c] for c in large_cls])
        for c in small_cls:
            nearest = large_cls[int(np.argmin(1.0 - (large_centroids @ centroids[c])))]
            louvain_labels[louvain_labels == c] = nearest
        re_map         = {old: new for new, old in enumerate(np.unique(louvain_labels))}
        louvain_labels = np.array([re_map[l] for l in louvain_labels])
        print(f"--> Merged {len(small_cls)} small clusters → còn {len(np.unique(louvain_labels))} clusters")
    else:
        print("--> Không có cluster nhỏ cần merge.")
    return louvain_labels


def evaluate(G, gt_labels, louvain_labels, final_partition, N):
    """Tính ARI, NMI, Modularity, Score."""
    ari        = adjusted_rand_score(gt_labels, louvain_labels)
    nmi        = normalized_mutual_info_score(gt_labels, louvain_labels,
                                              average_method="arithmetic")
    part_dict  = {i: int(louvain_labels[i]) for i in range(N)}
    modularity = community_louvain.modularity(part_dict, G, weight="weight")
    score      = compute_score(ari, nmi)
    return ari, nmi, modularity, score


def print_top5_clusters(louvain_labels, english_labels):
    clusters = {}
    for node_idx, cluster_id in enumerate(louvain_labels):
        clusters.setdefault(int(cluster_id), []).append(node_idx)
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
    print("--- CHI TIẾT TOP 5 CỤM LỚN NHẤT ---")
    for i, comm in enumerate(sorted_clusters[:5]):
        lbl_in_cluster   = [english_labels[n] for n in comm]
        top_label, top_c = Counter(lbl_in_cluster).most_common(1)[0]
        purity           = (top_c / len(comm)) * 100
        print(f"Cụm {i+1} ({len(comm)} ảnh) -> Nhãn đa số: {top_label} "
              f"(Purity: {purity:.1f}%) | ID đại diện: {list(comm)[:5]}")
    print("-" * 80)


def get_raw_image(ds, idx):
    return ds[int(idx)]["image"].convert("RGB")


def make_cluster_grid(ds, english_labels, sample_ids, cluster_id, cluster_label_names):
    rng    = np.random.default_rng(seed=cluster_id)
    chosen = rng.choice(sample_ids, size=min(IMGS_PER_CLUSTER, len(sample_ids)), replace=False)
    cols   = GRID_COLS
    rows   = int(np.ceil(len(chosen) / cols))
    thumb  = THUMB_SIZE
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols*(thumb/72)+.5, rows*(thumb/72)+1.2),
                             squeeze=False)
    fig.patch.set_facecolor("#1a1a2e")
    top_class = pd.Series(cluster_label_names).value_counts().index[0]
    n_unique  = len(set(cluster_label_names))
    fig.suptitle(f"Cluster {cluster_id}  │  {len(sample_ids)} ảnh  │  "
                 f"{n_unique} classes  │  top: {top_class}",
                 color="white", fontsize=9, fontweight="bold", y=0.99)
    for ax in axes.flat:
        ax.axis("off"); ax.set_facecolor("#1a1a2e")
    for plot_idx, sid in enumerate(chosen):
        r, c = divmod(plot_idx, cols)
        ax   = axes[r][c]
        img  = get_raw_image(ds, sid)
        img.thumbnail((thumb, thumb))
        ax.imshow(img)
        lbl = (english_labels[sid][:18] + "…") if len(english_labels[sid]) > 18 else english_labels[sid]
        ax.set_title(lbl, fontsize=5, color="#e0e0e0", pad=1)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def export_clusters_to_folders(louvain_labels, english_labels, ds,
                                cluster_to_ids, sorted_clusters, output_folder):
    """Lưu ảnh gốc vào từng folder theo cluster (đánh STT theo kích thước giảm dần)."""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    for rank_idx, cid in enumerate(tqdm(sorted_clusters, desc="  Lưu ảnh"), start=1):
        ids         = cluster_to_ids[cid]
        cluster_dir = os.path.join(output_folder, f"{rank_idx:03d}_cluster_{cid:04d}_size_{len(ids)}")
        os.makedirs(cluster_dir, exist_ok=True)
        for idx, sid in enumerate(ids, start=1):
            try:
                img        = get_raw_image(ds, sid)
                safe_label = str(english_labels[sid]).replace("/", "_").replace(" ", "_")
                img.save(os.path.join(cluster_dir, f"{idx:04d}_id{sid:05d}_{safe_label}.jpg"),
                         format="JPEG")
            except Exception as e:
                print(f"  [LỖI] ảnh {sid} cluster {cid}: {e}")
    print(f"--> Đã lưu {len(sorted_clusters)} cụm ảnh vào '{output_folder}'.")


def plot_size_distribution(cluster_to_ids, sorted_clusters, n_clusters,
                           ari, nmi, modularity, filename):
    sizes = [len(cluster_to_ids[c]) for c in sorted_clusters]
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.bar(range(len(sizes)), sizes,
           color=plt.cm.viridis(np.linspace(0.2, 0.85, len(sizes))))
    ax.set_xlabel("Cluster (rank by size)", color="white")
    ax.set_ylabel("Số ảnh",                 color="white")
    ax.set_title(f"Phân phối kích thước cluster  │  {n_clusters} clusters  │  "
                 f"ARI={ari:.3f}  NMI={nmi:.3f}  Mod={modularity:.3f}",
                 color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    plt.tight_layout()
    fig.savefig(filename, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"--> Size distribution → '{filename}'.")


def _run_one_config(feat, N, gt_labels, english_labels, ds,
                    k_neighbors, threshold, mutual_knn,
                    output_folder, viz_file, phase_label):
    """Pipeline hoàn chỉnh cho 1 bộ tham số. Trả về dict kết quả."""
    # Graph
    G, t_graph = build_knn_graph(feat, N, k_neighbors, threshold, mutual_knn)

    # Auto-tune resolution
    resolution = 1.0
    if AUTO_TUNE_RESOLUTION:
        resolution = auto_tune_resolution(G, len(np.unique(gt_labels)))
    else:
        print(f"--> Dùng Resolution={resolution} (không auto-tune)")

    # Multi-seed Louvain
    louvain_labels, final_partition, t_louvain = run_multi_seed_louvain(G, resolution, N)
    n_raw = len(np.unique(louvain_labels))

    # Merge small clusters
    louvain_labels = merge_small_clusters(louvain_labels, feat)
    n_clusters     = len(np.unique(louvain_labels))

    # Evaluate
    ari, nmi, modularity, score = evaluate(G, gt_labels, louvain_labels, final_partition, N)

    # Build cluster maps for export
    cluster_to_ids    = {}
    cluster_to_labels = {}
    for sid, cid in enumerate(louvain_labels):
        cluster_to_ids.setdefault(int(cid), []).append(sid)
        cluster_to_labels.setdefault(int(cid), []).append(english_labels[sid])
    sorted_clusters = sorted(cluster_to_ids.keys(),
                             key=lambda c: len(cluster_to_ids[c]), reverse=True)

    print(f"\n[{phase_label} RESULT]")
    print(f"  Params    : k={k_neighbors}, threshold={threshold}, mutual={mutual_knn}, "
          f"resolution={resolution:.4f}")
    print(f"  Graph     : {G.number_of_edges():,} cạnh  |  Build: {t_graph:.2f}s")
    print(f"  Clustering: {t_louvain:.2f}s  ({n_raw} → {n_clusters} clusters after merge)")
    print(f"  Metrics   : ARI={ari:.4f} | NMI={nmi:.4f} | "
          f"Modularity={modularity:.4f} | Score={score:.4f}")
    print("="*80)

    print_top5_clusters(louvain_labels, english_labels)

    # Export
    t_io = time.time()
    if SAVE_INDIVIDUAL_IMGS:
        export_clusters_to_folders(louvain_labels, english_labels, ds,
                                   cluster_to_ids, sorted_clusters, output_folder)
    t_io_elapsed = time.time() - t_io

    # Grid images for large clusters
    IMGS_DIR = os.path.join(OUTPUT_DIR, "cluster_images")
    os.makedirs(IMGS_DIR, exist_ok=True)
    for cid in sorted_clusters:
        if len(cluster_to_ids[cid]) < MAX_CLUSTER_IMGS:
            continue
        fig = make_cluster_grid(ds, english_labels, cluster_to_ids[cid],
                                cid, cluster_to_labels[cid])
        fig.savefig(os.path.join(IMGS_DIR, f"{phase_label.lower()}_cluster_{cid:04d}_grid.png"),
                    dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    # Size distribution
    t_viz = time.time()
    plot_size_distribution(cluster_to_ids, sorted_clusters, n_clusters,
                           ari, nmi, modularity, viz_file)
    t_viz_elapsed = time.time() - t_viz

    print(f"--> Xuất ảnh: {t_io_elapsed:.2f}s  |  Visualization: {t_viz_elapsed:.2f}s")

    return {
        "k_neighbors": k_neighbors, "threshold": threshold, "mutual_knn": mutual_knn,
        "resolution": resolution, "n_clusters": n_clusters,
        "ari": ari, "nmi": nmi, "modularity": modularity, "score": score,
        "labels": louvain_labels, "graph_edges": G.number_of_edges(),
        "t_louvain": t_louvain,
    }


# =============================================================================
# PHASE 1: BASELINE
# =============================================================================
def run_baseline(feat, N, gt_labels, english_labels, ds):
    print("\n" + "="*80)
    print("PHASE 1: BASELINE")
    print("="*80)
    t_start = time.time()

    result = _run_one_config(
        feat, N, gt_labels, english_labels, ds,
        k_neighbors  = BASELINE_K_NEIGHBORS,
        threshold    = BASELINE_THRESHOLD,
        mutual_knn   = BASELINE_MUTUAL_KNN,
        output_folder= BASELINE_FOLDER,
        viz_file     = BASELINE_VIZ_FILE,
        phase_label  = "BASELINE",
    )

    np.save('louvain_baseline_labels.npy', result["labels"])
    print(f"--> Tổng thời gian PHASE 1: {time.time()-t_start:.2f}s")
    return result


# =============================================================================
# PHASE 2: FULL TUNING
# =============================================================================
def run_tuning(feat, N, gt_labels, english_labels, ds):
    total_runs = (len(TUNING_K_NEIGHBORS_LIST) *
                  len(TUNING_THRESHOLD_LIST)   *
                  len(TUNING_MUTUAL_KNN_LIST))

    print("\n" + "="*80)
    print("PHASE 2: FULL TUNING")
    print(f"  Tổng số cấu hình chạy thử: {total_runs}")
    print("="*80)
    t_start    = time.time()
    all_results = []
    best        = {"score": -1}
    run_count   = 0

    for mutual in TUNING_MUTUAL_KNN_LIST:
        for k in TUNING_K_NEIGHBORS_LIST:
            # Xây graph 1 lần cho mỗi (mutual, k) — dùng lại qua các threshold
            # Nhưng threshold ảnh hưởng đến graph nên phải build riêng từng combo
            for thr in TUNING_THRESHOLD_LIST:
                run_count += 1
                print(f"\n  [{run_count:>2}/{total_runs}] "
                      f"k={k}, threshold={thr}, mutual={mutual}")
                print("  " + "-"*60)

                # Build graph
                G, t_graph = build_knn_graph(feat, N, k, thr, mutual)

                # Auto-tune resolution
                resolution = 1.0
                if AUTO_TUNE_RESOLUTION:
                    resolution = auto_tune_resolution(G, len(np.unique(gt_labels)))

                # Multi-seed Louvain
                louvain_labels, final_partition, t_louvain = \
                    run_multi_seed_louvain(G, resolution, N)

                # Merge small clusters
                louvain_labels = merge_small_clusters(louvain_labels, feat)
                n_clusters     = len(np.unique(louvain_labels))

                # Evaluate
                ari, nmi, modularity, score = evaluate(
                    G, gt_labels, louvain_labels, final_partition, N)

                print(f"  --> ARI={ari:.4f} | NMI={nmi:.4f} | "
                      f"Modularity={modularity:.4f} | Score={score:.4f} "
                      f"({t_louvain:.1f}s)")

                result = {
                    "k_neighbors": k, "threshold": thr, "mutual_knn": mutual,
                    "resolution": resolution, "n_clusters": n_clusters,
                    "ari": ari, "nmi": nmi, "modularity": modularity, "score": score,
                    "labels": louvain_labels, "graph_edges": G.number_of_edges(),
                    "t_louvain": t_louvain,
                }
                all_results.append(result)

                if score > best["score"]:
                    best = result

    # Bảng TOP 5
    print("\n" + "="*100)
    print("TOP 5 CẤU HÌNH TỐT NHẤT")
    print(f"{'k':>5} | {'thr':>5} | {'mutual':>6} | {'res':>7} | "
          f"{'n_cls':>6} | {'ARI':>8} | {'NMI':>8} | {'Score':>8} | "
          f"{'Mod':>7} | {'edges':>8} | {'time':>6}")
    print("-"*100)
    for r in sorted(all_results, key=lambda x: -x['score'])[:5]:
        is_best = (r['k_neighbors'] == best['k_neighbors'] and
                   r['threshold']   == best['threshold']   and
                   r['mutual_knn']  == best['mutual_knn'])
        marker = " ◄ BEST" if is_best else ""
        print(f"{r['k_neighbors']:>5} | {r['threshold']:>5} | {str(r['mutual_knn']):>6} | "
              f"{r['resolution']:>7.3f} | {r['n_clusters']:>6} | "
              f"{r['ari']:>8.4f} | {r['nmi']:>8.4f} | {r['score']:>8.4f} | "
              f"{r['modularity']:>7.4f} | {r['graph_edges']:>8,} | "
              f"{r['t_louvain']:>5.1f}s{marker}")
    print("="*100)

    print(f"\n[BEST TUNED CONFIGURATION]")
    print(f"  Params  : k={best['k_neighbors']}, threshold={best['threshold']}, "
          f"mutual={best['mutual_knn']}, resolution={best['resolution']:.4f}")
    print(f"  Graph   : {best['graph_edges']:,} cạnh")
    print(f"  Metrics : ARI={best['ari']:.4f} | NMI={best['nmi']:.4f} | "
          f"Modularity={best['modularity']:.4f} | Score={best['score']:.4f}")
    print("="*100)

    print_top5_clusters(best['labels'], english_labels)

    np.save('louvain_tuned_labels.npy', best['labels'])

    t_io = time.time()
    if SAVE_INDIVIDUAL_IMGS:
        # Build cluster maps for best result
        cluster_to_ids    = {}
        cluster_to_labels = {}
        for sid, cid in enumerate(best['labels']):
            cluster_to_ids.setdefault(int(cid), []).append(sid)
            cluster_to_labels.setdefault(int(cid), []).append(english_labels[sid])
        sorted_clusters = sorted(cluster_to_ids.keys(),
                                 key=lambda c: len(cluster_to_ids[c]), reverse=True)
        export_clusters_to_folders(best['labels'], english_labels, ds,
                                   cluster_to_ids, sorted_clusters, TUNING_FOLDER)

        IMGS_DIR = os.path.join(OUTPUT_DIR, "cluster_images")
        os.makedirs(IMGS_DIR, exist_ok=True)
        for cid in sorted_clusters:
            if len(cluster_to_ids[cid]) < MAX_CLUSTER_IMGS:
                continue
            fig = make_cluster_grid(ds, english_labels, cluster_to_ids[cid],
                                    cid, cluster_to_labels[cid])
            fig.savefig(os.path.join(IMGS_DIR, f"tuned_cluster_{cid:04d}_grid.png"),
                        dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

        plot_size_distribution(cluster_to_ids, sorted_clusters,
                               best['n_clusters'], best['ari'],
                               best['nmi'], best['modularity'], TUNING_VIZ_FILE)

    print(f"--> Tổng thời gian PHASE 2: {time.time()-t_start:.2f}s  "
          f"(Xuất ảnh: {time.time()-t_io:.2f}s)")
    return best


# =============================================================================
# MAIN
# =============================================================================
def main():
    t_total_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "cluster_images"), exist_ok=True)

    print(f"[INFO] Features  : {PRECOMPUTED_FEATURES_PATH}")
    print(f"[INFO] PCA={USE_PCA} (dim={PCA_DIM}, whiten={PCA_WHITEN})")
    print(f"[INFO] AutoTune={AUTO_TUNE_RESOLUTION}, N_seeds={N_SEEDS}, Alpha={ALPHA}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("BƯỚC 1: NẠP DỮ LIỆU")
    print("="*80)
    features, gt_raw, gt_labels, english_labels, ds, N, n_gt_classes = \
        load_features_and_dataset()

    if features.shape[0] != N:
        print(f"Lỗi: Số mẫu features ({features.shape[0]}) ≠ dataset ({N}).")
        return

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("BƯỚC 2: TIỀN XỬ LÝ FEATURES")
    print("="*80)
    feat_norm = preprocess_features(features)

    # ── 3. Phase 1 & 2 ───────────────────────────────────────────────────────
    baseline_result = run_baseline(feat_norm, N, gt_labels, english_labels, ds)
    tuned_result    = run_tuning(feat_norm, N, gt_labels, english_labels, ds)

    # ── 4. So sánh cuối ───────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"SO SÁNH BASELINE vs BEST TUNED  (Score = {ALPHA:.1f}×ARI + {1-ALPHA:.1f}×NMI)")
    print(f"{'':25} {'BASELINE':>12} {'BEST TUNED':>12} {'DELTA':>10}")
    print("-"*65)
    for label, bkey in [("k_neighbors", "k_neighbors"),
                         ("threshold",   "threshold"),
                         ("mutual_knn",  "mutual_knn"),
                         ("resolution",  "resolution"),
                         ("n_clusters",  "n_clusters")]:
        bv = str(baseline_result[bkey])
        tv = str(tuned_result[bkey])
        print(f"  {label:<22} {bv:>12} {tv:>12}")
    print("-"*65)
    for label, key in [("ARI", "ari"), ("NMI", "nmi"),
                        ("Modularity", "modularity"), ("Score", "score")]:
        bv    = baseline_result[key]
        tv    = tuned_result[key]
        delta = tv - bv
        print(f"  {label:<22} {bv:>12.4f} {tv:>12.4f} {delta:>+10.4f}")
    print("="*80)

    # ── 5. Tổng thời gian ────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total_start
    h = int(total_elapsed // 3600)
    m = int((total_elapsed % 3600) // 60)
    s = total_elapsed % 60
    time_str = (f"{h} giờ {m} phút {s:.2f} giây" if h > 0 else
                f"{m} phút {s:.2f} giây"          if m > 0 else
                f"{s:.2f} giây")
    print(f"\n--> TỔNG THỜI GIAN CHẠY TOÀN BỘ CHƯƠNG TRÌNH: {time_str}")


if __name__ == "__main__":
    main()
