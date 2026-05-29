import numpy as np
import igraph as ig
import leidenalg as la
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
import json
import os


# Config 
FEATURES_PATH  = "features.npy"
LABELS_PATH    = "labels.npy"
SAVE_DIR       = "."
RANDOM_SEED    = 42
N_LEIDEN_ITER  = 10

# Grid search space 
K_LIST         = [15, 20, 25, 30, 40, 50, 60, 70, 80]
THRESHOLD_LIST = [0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6] 

PARTITION_CONFIGS = [
    (la.CPMVertexPartition, [0.01, 0.02, 0.03, 0.05, 0.07, 0.10], "CPM"),
]

X      = np.load(FEATURES_PATH).astype(np.float32)
y_true = np.load(LABELS_PATH)
N      = X.shape[0]
print(f"      Feature matrix : {X.shape}")
print(f"      Label count    : {len(np.unique(y_true))} unique classes")

def compute_knn(X, k, metric="cosine"):
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    return distances, indices


def build_mutual_knn_graph(k, threshold, distances, indices):

    neighbor_sets = [set(row) for row in indices]

    edges, weights = [], []
    for i in range(N):
        for j_pos in range(1, k):
            j = indices[i, j_pos]
            if j <= i:                      
                continue
            if i not in neighbor_sets[j]:    
                continue

            sim = 1.0 - distances[i, j_pos]

            if sim < threshold:
                continue

            shared  = len(neighbor_sets[i] & neighbor_sets[j])
            union   = len(neighbor_sets[i] | neighbor_sets[j])
            jaccard = shared / union if union > 0 else 0.0

            w = sim * (1.0 + jaccard)
            edges.append((i, j))
            weights.append(w)

    if not edges:
        return None

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights
    g.simplify(combine_edges="mean")
    return g

def run_leiden(g, partition_class, resolution, seed, n_iterations):
    kwargs = dict(weights="weight", seed=seed, n_iterations=n_iterations)
    if resolution is not None:
        kwargs["resolution_parameter"] = resolution

    partition = la.find_partition(g, partition_class, **kwargs)
    return np.array(partition.membership)


def evaluate(y_true, y_pred):
    ari        = adjusted_rand_score(y_true, y_pred)
    nmi        = normalized_mutual_info_score(y_true, y_pred,
                                              average_method="arithmetic")
    n_clusters = len(set(y_pred))
    return ari, nmi, n_clusters

print("\n Starting grid search …")
print(f"      k values        : {K_LIST}")
print(f"      thresholds      : {THRESHOLD_LIST}")
print(f"      partition types : {[cfg[2] for cfg in PARTITION_CONFIGS]}")
total_start = time.time()

results     = []
best_ari    = -1.0
best_score = 0
best_params = {}

for k in K_LIST:
    print(f"\n{'='*64}")
    print(f"  Computing kNN for k={k} …")
    knn_start          = time.time()
    distances, indices = compute_knn(X, k)
    print(f"  kNN computed in {time.time()-knn_start:.1f}s")

    local_thresh = 1.0 - distances[:, 1:].mean(axis=1) 

    for threshold in THRESHOLD_LIST:
        g = build_mutual_knn_graph(
            k, threshold, distances, indices)

        if g is None:
            print(f"  k={k} thresh={threshold} → SKIP (no edges)")
            continue

        n_edges     = g.ecount()
        n_connected = sum(1 for v in g.vs if v.degree() > 0)
        print(f"\n  k={k} thresh={threshold} | "
                f"edges={n_edges:,} | connected={n_connected}/{N}")

        if n_edges < 100:
            print(f"  → SKIP (too sparse)")
            continue

        for partition_class, resolution_list, partition_label in PARTITION_CONFIGS:
            for resolution in resolution_list:

                try:
                    y_pred = run_leiden(
                        g,
                        partition_class,
                        resolution,
                        seed         = RANDOM_SEED,
                        n_iterations = N_LEIDEN_ITER,
                    )
                except Exception as e:
                    print(f"    [{partition_label} res={resolution}] ERROR: {e}")
                    continue

                ari, nmi, n_clusters = evaluate(y_true, y_pred)
                score = 0.8 * ari + 0.2 * nmi

                res_str = f"{resolution:.3f}" if resolution is not None else "auto"
                is_best = score > best_score
                print(
                    f"    [{partition_label:12s} res={res_str:5s}] "
                    f"clusters={n_clusters:<5d} "
                    f"ARI={ari:.4f}  NMI={nmi:.4f}"
                    + (" ← BEST ARI" if is_best else "")
                )

                record = {
                    "k"            : k,
                    "threshold"    : threshold,
                    "partition"    : partition_label,
                    "resolution"   : resolution,
                    "n_clusters"   : n_clusters,
                    "n_edges"      : n_edges,
                    "ari"          : round(ari, 6),
                    "nmi"          : round(nmi, 6),
                    "score"        : round(score, 6),
                }
                results.append(record)

                if is_best:
                    best_score    = score
                    best_params = record.copy()

# Summary 
elapsed = time.time() - total_start
print(f"\n\n{'='*64}")
print(f"  GRID SEARCH COMPLETE  ({elapsed:.1f}s, {len(results)} configs evaluated)")
print(f"{'='*64}")
print(f"  Best configuration:")
print(f"    k                  : {best_params['k']}")
print(f"    threshold          : {best_params['threshold']}")
print(f"    partition          : {best_params['partition']}")
print(f"    resolution         : {best_params['resolution']}")
print(f"    n_clusters         : {best_params['n_clusters']}")
print(f"    ARI                : {best_params['ari']:.4f}")
print(f"    NMI                : {best_params['nmi']:.4f}")
print(f"{'='*64}")

# Save all results 

results_path = os.path.join(SAVE_DIR, "grid_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

best_k         = best_params["k"]
best_threshold = best_params["threshold"]
best_partition = next(
    cfg for cfg in PARTITION_CONFIGS if cfg[2] == best_params["partition"]
)
best_resolution = best_params["resolution"]

distances_best, indices_best = compute_knn(X, best_k)

g_best = build_mutual_knn_graph(best_k, best_threshold, distances_best, indices_best)

y_final = run_leiden(
    g_best,
    best_partition[0],
    best_resolution,
    seed         = RANDOM_SEED,
    n_iterations = N_LEIDEN_ITER * 2,  
)

ari_final, nmi_final, nc_final = evaluate(y_true, y_final)
print(f"      Final  ARI={ari_final:.4f}  NMI={nmi_final:.4f}  clusters={nc_final}")

np.save(os.path.join(SAVE_DIR, "labels_pred.npy"), y_final)
print(f"      Saved final predictions → labels_pred.npy")
