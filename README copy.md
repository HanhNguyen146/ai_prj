# Spectral Clustering for Community Structure Identification on ImageNet-Hard

This repository implements **Spectral Clustering (SC)** as one of four algorithms used to identify community structures in the ImageNet-Hard dataset. The other algorithms in the study are **Leiden**, **DBSCAN**, and **Louvain**.

## Overview

The goal is to group images from ImageNet-Hard into meaningful communities (clusters) based on their visual feature representations, then evaluate how well these clusters align with the true class labels.

## Dataset

**ImageNet-Hard** (`taesiri/imagenet-hard` on Hugging Face) — a benchmark of images that consistently fool state-of-the-art vision models, making it a challenging testbed for community detection.

## Method

The pipeline has two phases:

**1. Baseline**
- Builds a k-NN similarity graph from pre-extracted image features using cosine distance
- Applies spectral clustering with fixed parameters (`n_neighbors=20`, `threshold=0.25`, `K=100`)

**2. Hyperparameter Tuning**
- Grid-searches 48 configurations across:
  - `n_neighbors`: 20, 30, 40
  - `threshold`: 0.1, 0.2, 0.25, 0.3
  - `K` (clusters): 200, 500, 700, 1000
- Selects the best configuration using a weighted score: `score = α × ARI + (1 − α) × NMI`

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| ARI | Adjusted Rand Index — clustering agreement with ground truth |
| NMI | Normalized Mutual Information — information overlap with true labels |

## Project Structure

```
DA_AI/
├── SC.py                          # Main spectral clustering pipeline
├── features_no_umap.npy           # Pre-extracted image feature vectors
├── spectral_baseline_labels.npy   # Cluster assignments (baseline)
├── spectral_tuned_labels.npy      # Cluster assignments (tuned)
├── SC_Baseline_Visualization.png  # t-SNE + graph plot (baseline)
├── SC_Tuned_Visualization.png     # t-SNE + graph plot (tuned)
├── Cluster_Results_SC_Baseline/   # Images organized by cluster (baseline)
├── Cluster_Results_SC_Tuned/      # Images organized by cluster (tuned)
└── ZoomIsAllYouNeed-main/         # Reference benchmark code
```

## Requirements

```
numpy
scikit-learn
scipy
networkx
matplotlib
datasets  # Hugging Face
Pillow
```

## Usage

```bash
python SC.py
```

This will run the baseline, then the full tuning sweep, and output:
- Cluster label files (`.npy`)
- Visualization plots (`.png`)
- Image folders organized by cluster

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALPHA` | 0.5 | Weight between ARI and NMI in scoring |
| `KNN_METRIC` | `cosine` | Distance metric for k-NN graph |
| `N_NODES_VISUALIZATION` | 300 | Nodes shown in graph visualization |
