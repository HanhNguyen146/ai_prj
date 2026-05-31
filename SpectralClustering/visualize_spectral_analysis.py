import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Cấu hình tham số tương tự SC.py
KNN_METRIC = 'cosine'
N_NEIGHBORS = 20
THRESHOLD = 0.25
N_EIGENVALUES = 1000  # Số trị riêng cần tính để vẽ biểu đồ phổ

print("--> Đang nạp features.npy...")
try:
    features = np.load('features.npy')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy features.npy tại thư mục làm việc!")
    exit(1)

print(f"--> Đang xây dựng ma trận tương đồng (KNN Graph)...")
nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric=KNN_METRIC, n_jobs=-1)
nbrs.fit(features)

distance_matrix = nbrs.kneighbors_graph(mode='distance')
similarity_matrix = distance_matrix.copy()
similarity_matrix.data = 1.0 - similarity_matrix.data
similarity_matrix.data[similarity_matrix.data <= THRESHOLD] = 0
similarity_matrix.eliminate_zeros()

# Đối xứng hóa
similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
similarity_matrix.eliminate_zeros()

print("--> Đang tính toán ma trận Normalized Laplacian L_sym...")
# Tính ma trận độ bậc D
degrees = np.array(similarity_matrix.sum(axis=1)).flatten()
degrees[degrees == 0] = 1e-12  # Tránh chia cho 0
d_inv_sqrt = 1.0 / np.sqrt(degrees)
D_inv_sqrt = sp.diags(d_inv_sqrt)

# L_sym = I - D^{-1/2} * W * D^{-1/2}
I = sp.eye(similarity_matrix.shape[0])
L_sym = I - D_inv_sqrt.dot(similarity_matrix).dot(D_inv_sqrt)

print(f"--> Đang thực hiện phân tách trị riêng để lấy {N_EIGENVALUES} trị riêng nhỏ nhất...")
# eigsh tìm các trị riêng nhỏ nhất (SM: Smallest Magnitude)
eigenvalues, eigenvectors = eigsh(L_sym, k=N_EIGENVALUES, which='SM')

# Sắp xếp trị riêng theo thứ tự tăng dần
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("--> Đang vẽ các biểu đồ trực quan hóa phổ...")

# Đảm bảo thư mục image tồn tại
import os
os.makedirs('image', exist_ok=True)

# 1. Vẽ biểu đồ phổ trị riêng (Eigenvalue Spectrum)
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, N_EIGENVALUES + 1), eigenvalues, 'o-', color='#3182ce', linewidth=2, markersize=6)
plt.title("Biểu đồ Phổ trị riêng (Eigenvalue Spectrum) của ma trận Laplacian", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Thứ tự Trị riêng (Eigenvalue Index)", fontsize=10)
plt.ylabel("Giá trị Trị riêng (Eigenvalue)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
# Đánh dấu Eigengap
# plt.annotate('Eigengap (Vùng nhảy vọt)', xy=(5, eigenvalues[4]), xytext=(10, eigenvalues[4] + 0.15),
            #  arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6))
plt.tight_layout()
plt.savefig('image/spectral_eigenvalues_spectrum.png', dpi=300)
plt.close()
print("--> Đã lưu biểu đồ phổ trị riêng vào 'image/spectral_eigenvalues_spectrum.png'")

# 2. Vẽ không gian nhúng phổ (Spectral Embedding Space) bằng Vector riêng thứ 2 và thứ 3
plt.figure(figsize=(8, 8))
plt.scatter(eigenvectors[:, 1], eigenvectors[:, 2], s=2, alpha=0.6, color='#2b6cb0')
plt.title("Không gian nhúng phổ (Spectral Embedding Space)\nTrực quan hóa bằng Vector riêng 2 & 3", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Vector riêng thứ 2 (2nd Eigenvector)", fontsize=10)
plt.ylabel("Vector riêng thứ 3 (3rd Eigenvector)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('image/spectral_embedding_space.png', dpi=300)
plt.close()
print("--> Đã lưu biểu đồ không gian nhúng phổ vào 'image/spectral_embedding_space.png'")

print("--> HOÀN TẤT!")