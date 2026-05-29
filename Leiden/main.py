import numpy as np
import igraph as ig
import leidenalg as la
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datasets import load_dataset
import random

FEATURES_PATH  = "features.npy"
LABELS_PATH    = "labels.npy"

BEST_K         = 50
BEST_THRESHOLD = 0.3
BEST_RES       = 0.01
N_LEIDEN_ITER  = 10
RANDOM_SEED    = 42

X      = np.load(FEATURES_PATH).astype(np.float32)
y_true = np.load(LABELS_PATH)
N      = X.shape[0]


# kNN 
def compute_knn(X, k):
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    nn.fit(X)
    return nn.kneighbors(X)

t_knn = time.time()

distances, indices = compute_knn(X, BEST_K)



# Build Graph 
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

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights
    g.simplify(combine_edges="mean")
    return g


g = build_mutual_knn_graph(BEST_K, BEST_THRESHOLD, distances, indices)

print(f"    Edges: {g.ecount():,}")


# Leiden 
def run_leiden(g):
    partition = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=BEST_RES,
        seed=RANDOM_SEED,
        n_iterations=N_LEIDEN_ITER,
    )
    return np.array(partition.membership)

t_leiden = time.time()

y_pred = run_leiden(g)
np.save("y_pred.npy", y_pred)

print(f"    Leiden time: {time.time() - t_knn:.2f}s")


ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)

print(f"    ARI = {ari:.4f}")
print(f"    NMI = {nmi:.4f}")

