import re
import matplotlib.pyplot as plt
import numpy as np

import os

log_file_path = "logSC.txt"
output_image_path = "image/spectral_tuning_sweeps.png"
os.makedirs("image", exist_ok=True)

# Regex để parse dòng log:
#  [ 1/48] n_nbrs=20, thr=0.1, K=200 -> ARI=0.1269 | NMI=0.6466 | Score=0.3868 (32.3s)
pattern = re.compile(
    r"n_nbrs=(\d+),\s+thr=([\d\.]+),\s+K=(\d+).+?ARI=([\d\.]+)\s+\|\s+NMI=([\d\.]+)\s+\|\s+Score=([\d\.]+)"
)

runs = []

print(f"--> Đang đọc và phân tích cú pháp {log_file_path}...")
try:
    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                n_nbrs = int(match.group(1))
                thr = float(match.group(2))
                k = int(match.group(3))
                ari = float(match.group(4))
                nmi = float(match.group(5))
                score = float(match.group(6))
                runs.append({
                    "n_neighbors": n_nbrs,
                    "threshold": thr,
                    "k": k,
                    "ari": ari,
                    "nmi": nmi,
                    "score": score
                })
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {log_file_path}!")
    exit(1)

print(f"--> Đã trích xuất thành công {len(runs)} cấu hình chạy thử.")

if len(runs) == 0:
    print("Lỗi: Không parse được cấu hình nào! Vui lòng kiểm tra lại cấu trúc file log.")
    exit(1)

# Gom nhóm dữ liệu theo (threshold, K) và lấy trung bình các n_neighbors
data_grouped = {}
for r in runs:
    key = (r["threshold"], r["k"])
    data_grouped.setdefault(key, []).append(r["score"])

thresholds = sorted(list(set(r["threshold"] for r in runs)))
k_values = sorted(list(set(r["k"] for r in runs)))

plt.figure(figsize=(10, 6))

colors = {0.1: "#3182ce", 0.2: "#38a169", 0.25: "#dd6b20", 0.3: "#e53e3e"}
markers = {0.1: "o", 0.2: "s", 0.25: "^", 0.3: "d"}

for thr in thresholds:
    scores = []
    for k in k_values:
        scores.append(np.mean(data_grouped[(thr, k)]))
    
    # Làm nổi bật đường threshold = 0.2 (cấu hình tốt nhất) bằng nét vẽ dày hơn
    linewidth = 3.0 if thr == 0.2 else 1.5
    label = f"Ngưỡng threshold = {thr}" + (" (Tối ưu)" if thr == 0.2 else "")
    
    plt.plot(k_values, scores, marker=markers[thr], color=colors[thr], 
             linewidth=linewidth, markersize=8, label=label)

plt.title("Tác động của Số lượng cụm (K) và Ngưỡng lọc cạnh (threshold) tới Composite Score", 
          fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Số lượng cụm kỳ vọng (K)", fontsize=10)
plt.ylabel("Composite Score (Trung bình cộng ARI & NMI)", fontsize=10)
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
plt.tight_layout()

plt.savefig(output_image_path, dpi=300)
plt.close()
print(f"--> Đã lưu đồ thị xu hướng tinh chỉnh vào '{output_image_path}'!")
print("--> HOÀN TẤT!")