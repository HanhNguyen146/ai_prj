# Xử lý bài toán Community Structure Identification sử dụng các thuật toán Khoa học Máy tính
## (Solving the Community Structure Identification problem using Computer Science algorithms)

Đồ án này tập trung vào việc nghiên cứu, áp dụng và so sánh **4 thuật toán phân cụm đồ thị/cộng đồng (Community Detection)** để gom nhóm hình ảnh của tập dữ liệu thách thức **ImageNet-Hard** dựa trên các đặc trưng ảnh được trích xuất bằng mô hình **CLIP**.

Các thuật toán được thử nghiệm bao gồm:
1. **Spectral Clustering (SC)**
2. **Louvain Algorithm**
3. **Leiden Algorithm**
4. **Infomap Algorithm**

---

## 📂 Tổ chức dự án (Project Structure)

Để đảm bảo tính độc lập và an toàn cho cấu hình của từng thuật toán, dự án được tổ chức thành 4 thư mục độc lập. Mỗi thuật toán có bộ tham số và thứ tự chạy file riêng:

```text
ai_prj/
├── requirements.txt                 # Khai báo tất cả thư viện dùng chung cho 4 thuật toán
├── README.md                        # Hướng dẫn chạy đồ án này
│
├── sc/                              # Thuật toán Spectral Clustering
│   ├── SC.py                        # Pipeline chạy chính (Baseline & Tuning)
│   ├── visualize_sc.py              # Trực quan hóa kết quả phân cụm
│   ├── visualize_spectral_analysis.py
│   ├── visualize_tuning.py
│   └── features.npy, features_no_umap.npy  # File đặc trưng trích xuất
│
├── louvain/                         # Thuật toán Louvain
│   ├── louvain_fixed.py             # Pipeline chạy chính (Baseline & Tuning)
│   ├── draw_visual.py               # Trực quan hóa
│   └── features.npy, features_no_umap.npy  # File đặc trưng trích xuất
│
├── leiden/                          # Thuật toán Leiden
│   ├── feature_extraction.py        # Trích xuất đặc trưng ảnh bằng mô hình CLIP
│   ├── grid_search.py               # Quét tìm siêu tham số tối ưu (k, threshold, resolution)
│   ├── main.py                      # Chạy phân cụm Leiden với cấu hình tốt nhất
│   └── visualize.py                 # Trực quan hóa cụm ảnh ngẫu nhiên
│
└── infomap/                         # Thuật toán Infomap
    ├── main.py                      # Chạy chính (Grid Search + Đánh giá + So sánh)
    ├── load.py                      # Hàm phụ trợ load dữ liệu
    └── features.npy                 # File đặc trưng trích xuất
```

---

## 🛠️ Cài đặt môi trường (Installation)

Trước khi chạy bất kỳ thuật toán nào, hãy cài đặt tất cả các thư viện cần thiết bằng cách mở Terminal tại thư mục gốc `ai_prj/` và chạy lệnh sau:

```bash
pip install -r requirements.txt
```

*Lưu ý: Môi trường yêu cầu Python 3.8+ và khuyến nghị sử dụng GPU (nếu chạy phần trích xuất đặc trưng bằng PyTorch trong thư mục `leiden/`).*

---

## 🚀 Hướng dẫn chạy từng thuật toán (Running Algorithms)

### 1. Spectral Clustering (Phân cụm phổ)
Di chuyển vào thư mục `sc/` và chạy chương trình chính:
```bash
cd sc
python SC.py
```
* **Mô tả:** File [SC.py](file:///c:/Hanh/GITHUB/ai_prj/sc/SC.py) sẽ chạy 2 giai đoạn: Giai đoạn Baseline (k-NN graph cố định) và Giai đoạn Tuning quét qua 48 cấu hình tham số. Kết quả sẽ lưu các ảnh t-SNE phân cụm và xuất các nhóm ảnh thực tế vào thư mục `Cluster_Results_SC_Baseline/` và `Cluster_Results_SC_Tuned/`.

### 2. Louvain Algorithm
Di chuyển vào thư mục `louvain/` và chạy chương trình chính:
```bash
cd louvain
python louvain_fixed.py
```
* **Mô tả:** Thực hiện thuật toán Louvain tìm kiếm cộng đồng tối ưu hóa độ đo Modularity. Bật tính năng tự động dò tìm resolution parameter để số cụm khớp với số nhãn gốc (ground-truth). Kết quả lưu tại folder `louvain_results_opt/`.

### 3. Leiden Algorithm
Thuật toán Leiden được chạy theo quy trình 3 bước để tối ưu hóa hiệu quả:

* **Bước 1 (Trích xuất đặc trưng):** Tạo file đặc trưng `features.npy` và `labels.npy` bằng CLIP (nếu chưa có sẵn):
  ```bash
  cd leiden
  python feature_extraction.py
  ```
* **Bước 2 (Tìm tham số tốt nhất):** Chạy Grid Search quét tìm cấu hình siêu tham số:
  ```bash
  python grid_search.py
  ```
  *(Quá trình này sẽ tìm ra cấu hình cho chỉ số ARI/NMI cao nhất và lưu vào file `grid_results.json`)*
* **Bước 3 (Chạy mô hình tối ưu & Trực quan hóa):** Chạy thuật toán Leiden trên cấu hình tốt nhất và vẽ biểu đồ khám phá:
  ```bash
  python main.py
  python visualize.py
  ```

### 4. Infomap Algorithm
Di chuyển vào thư mục `infomap/` và chạy chương trình:
```bash
cd infomap
python main.py
```
* **Mô tả:** Chạy quét tham số (k-NN Neighbors và Markov Time) trên tập huấn luyện (Train), đánh giá độ chính xác phân cụm bằng thuật toán đối xạ nhãn Hungarian trên tập kiểm thử (Test), vẽ so sánh side-by-side kết quả dự đoán với Ground Truth và khám phá ngữ nghĩa hình ảnh.

---

## 📊 Chỉ số Đánh giá & So sánh (Evaluation Metrics)

Để đánh giá và đưa vào báo cáo đồ án, các thuật toán đều sử dụng chung các độ đo tiêu chuẩn:
* **ARI (Adjusted Rand Index):** Đo lường độ tương đồng giữa phân cụm dự đoán và nhãn gốc (Ground Truth).
* **NMI (Normalized Mutual Information):** Đo lường lượng thông tin chung giữa phân cụm và nhãn gốc.
* **Purity:** Độ tinh khiết của các cụm được phát hiện.
* **Execution Time (Thời gian chạy):** Thời gian dựng đồ thị k-NN và thời gian phân cụm của thuật toán.

---

## ⚠️ Lưu ý quan trọng khi nộp bài và Push lên GitHub
1. **Tránh lỗi "thư mục xám":** Dự án này đã được kiểm tra và chỉ sử dụng một cấu hình Git duy nhất tại thư mục gốc. **Không** khởi tạo thêm lệnh `git init` bên trong các thư mục con (`sc`, `louvain`, `leiden`, `infomap`) để tránh làm lỗi hiển thị mã nguồn trên GitHub.
2. **File dung lượng lớn (.npy):** Khuyến khích thêm các file `.npy` dung lượng lớn vào tệp `.gitignore` khi nộp lên GitHub để tránh vượt quá giới hạn file của GitHub (100MB) và giúp đẩy code nhanh hơn.
