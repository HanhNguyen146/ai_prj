import numpy as np
import igraph as ig
import leidenalg as la
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
import matplotlib.pyplot as plt
from datasets import load_dataset
import random
import os

FEATURES_PATH  = "features.npy"
LABELS_PATH    = "labels.npy"
PRED_LABELS_PATH = "y_pred.npy"

y_true = np.load(LABELS_PATH)
y_pred = np.load(PRED_LABELS_PATH)

ds_images = load_dataset("taesiri/imagenet-hard", split="validation")

def plot_cluster_image_grid(y_pred, num_clusters_to_show=4, images_per_cluster=6):
    unique_clusters = list(set(y_pred))
    
    valid_clusters = [c for c in unique_clusters if list(y_pred).count(c) >= images_per_cluster]

    sampled_clusters = random.sample(valid_clusters, min(num_clusters_to_show, len(valid_clusters)))

    fig, axes = plt.subplots(num_clusters_to_show, images_per_cluster, figsize=(16, 3 * num_clusters_to_show))
    fig.suptitle("Khám phá ngữ nghĩa các cụm ảnh do thuật toán tìm ra", fontsize=18, fontweight='bold', y=0.98)

    os.makedirs("output", exist_ok=True)

    for row, cluster_id in enumerate(sampled_clusters):
        indices_in_cluster = np.where(y_pred == cluster_id)[0]

        sampled_indices = random.sample(list(indices_in_cluster), images_per_cluster)

        for col, idx in enumerate(sampled_indices):
            img = ds_images[int(idx)]["image"]
            true_label_id = y_true[idx]
            
            raw_eng_labels = ds_images[int(idx)]["english_label"]
            
            if len(raw_eng_labels) > 0:
                eng_label = raw_eng_labels[0].split(',')[0][:20] 
            else:
                eng_label = "Unknown"

            ax = axes[row, col] if num_clusters_to_show > 1 else axes[col]
            ax.imshow(img)
            ax.axis('off')
            
            if col == 0:
                ax.set_title(f"Cụm: {cluster_id}\nGốc: {true_label_id} ({eng_label})", 
                             loc='left', fontsize=11, fontweight='bold', color='darkred')
            else:
                ax.set_title(f"Gốc: {true_label_id} ({eng_label})", 
                             fontsize=10, color='blue')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("output/cluster_images_grid_3.png", dpi=300)

plot_cluster_image_grid(y_pred, num_clusters_to_show=4, images_per_cluster=6)
