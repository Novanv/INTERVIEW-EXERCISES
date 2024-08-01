# Phân loại MNIST với Triplet Loss

Dự án này triển khai một mô hình học sâu để phân loại ký tự quang học sử dụng tập dữ liệu MNIST, áp dụng hàm mất mát Triplet Loss trong quá trình huấn luyện. Mô hình được triển khai chỉ với NumPy, với mục tiêu cung cấp sự hiểu biết trực tiếp về mạng nơ-ron và triplet loss.

## Cài đặt

1. **Cài đặt các Thư viện Cần Thiết**: Đảm bảo bạn đã cài đặt các thư viện Python cần thiết. Bạn có thể sử dụng `pip` hoặc `conda` để quản lý gói. Dự án này yêu cầu NumPy, SciPy và scikit-learn.

    ```bash
    pip install numpy scipy scikit-learn matplotlib
    ```

2. **Chuẩn bị Dữ Liệu**: Đặt tệp dữ liệu MNIST của bạn (`mnist_784.arff`) vào thư mục `data/`.

Bạn có thể tải xuống tập dữ liệu MNIST từ OpenML bằng liên kết sau:

[Tải xuống Tập dữ liệu MNIST](https://www.openml.org/search?type=data&sort=runs&id=554&status=active)

Tập dữ liệu này được sử dụng cho việc huấn luyện và kiểm tra.

3. **Chạy Dự Án**: Thực thi tập lệnh chính để chạy toàn bộ quy trình.

    ```bash
    python main.py
    ```

## Giải thích Phương Pháp

1. **Tải và Tiền Xử Lý Dữ Liệu**:
   - Tập dữ liệu MNIST được tải từ tệp ARFF và phân chia thành tập huấn luyện và tập kiểm tra.
   - Dữ liệu được chuẩn hóa để đảm bảo tỉ lệ đầu vào nhất quán.

2. **Kiến Trúc Mô Hình**:
   - Sử dụng một mạng nơ-ron đơn giản với một lớp ẩn để tạo ra các embedding.
   - Phép truyền trực tiếp áp dụng kích hoạt ReLU và biến đổi tuyến tính để tạo ra các embedding.

3. **Hàm Mất Mát Triplet**:
   - Triplet Loss nhằm đảm bảo rằng một ví dụ anchor gần gũi hơn với các ví dụ dương (cùng lớp) hơn là các ví dụ âm (khác lớp) theo một biên nhất định.
   - Các triplet được tạo ra trong quá trình huấn luyện, và hàm mất mát phạt các embedding không đáp ứng các ràng buộc về sự tương tự mong muốn.

4. **Huấn luyện và Đánh giá**:
   - Mô hình được huấn luyện sử dụng hàm mất mát Triplet Loss và các embedding được học.
   - Để đánh giá, các embedding được sử dụng với bộ phân loại k-Nearest Neighbors (k-NN) để đánh giá độ chính xác phân loại.

## Ưu điểm của Phương Pháp Học Sâu với Triplet Loss

1. **Học Đặc Trưng**: Các mô hình học sâu, đặc biệt là với triplet loss, có thể học được các đại diện đặc trưng có ý nghĩa, từ đó có thể dẫn đến khả năng tổng quát tốt hơn trên dữ liệu chưa thấy.
2. **Tính Linh Hoạt**: Mạng nơ-ron có thể dễ dàng mở rộng với nhiều lớp hoặc kiến trúc khác nhau để cải thiện hiệu suất.
3. **Khả Năng Chịu Đựng Tốt**: Triplet Loss có thể chịu đựng tốt hơn đối với dữ liệu nhiễu và biến thể so với các phương pháp phân loại truyền thống.

## Nhược điểm

1. **Độ Phức Tạp**: Huấn luyện các mô hình học sâu có thể tốn kém về tính toán và yêu cầu nhiều tài nguyên hơn so với các mô hình học máy đơn giản hơn.
2. **Khai Thác Triplet**: Việc tạo ra các triplet hiệu quả là rất quan trọng và có thể gặp khó khăn. Việc chọn triplet kém có thể ảnh hưởng tiêu cực đến hiệu suất mô hình.
3. **Rủi Ro Overfitting**: Với các mô hình phức tạp hơn, có nguy cơ overfitting, đặc biệt nếu tập dữ liệu không đủ lớn hoặc đa dạng.

## So sánh với Phương Pháp Machine Learning Trước Đó

- **Phương Pháp Machine Learning**: Phương pháp trước đó sử dụng Logistic Regression với chiến lược one-vs-all. Phương pháp này đơn giản và hiệu quả cho các tập dữ liệu nhỏ nhưng có thể gặp khó khăn với các mẫu dữ liệu phức tạp hơn.
- **Phương Pháp Học Sâu**: Phương pháp học sâu với triplet loss có thể cung cấp hiệu suất tốt hơn cho các bài toán phức tạp hơn bằng cách học các đại diện đặc trưng phong phú hơn. Tuy nhiên, nó yêu cầu quản lý các siêu tham số cẩn thận và có thể tiêu tốn nhiều tài nguyên hơn.

## Cấu Trúc Dự Án
- `config.py`: Chứa các tham số cấu hình cho dự án.
- `data`: Chứa tập dữ liệu MNIST ở định dạng ARFF.
- `utils.py`: Xử lý việc tải dữ liệu, tiền xử lý và trực quan hóa, triển khai hàm mất mát Triplet Loss và tạo triplet.
- `model.py`: Định nghĩa kiến trúc mạng nơ-ron và quy trình huấn luyện.
- `inference.py`: huấn luyện và đánh giá mô hình
- `main.py`: Tập lệnh chính tích hợp toàn bộ quy trình.

## Acknowledgments

- Tập dữ liệu MNIST được cung cấp bởi Yann LeCun và AT&T Labs.
- Xin cảm ơn các cộng tác viên và thư viện đã làm cho dự án này trở nên khả thi.