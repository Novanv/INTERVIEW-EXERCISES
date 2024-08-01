# MNIST Classification API

## 1. Giới thiệu

API này cung cấp dịch vụ phân loại hình ảnh MNIST bằng cách sử dụng một mô hình mạng nơ-ron đã được huấn luyện kết hợp với k-NN classifier để phân loại các hình ảnh chữ số viết tay. Dưới đây là hướng dẫn cách sử dụng API.

## 2. Cấu trúc dự án
- `templates/index.html`: Tệp HTML cho giao diện người dùng của API.
- `data`: Chứa tập dữ liệu MNIST ở định dạng ARFF.
- `weighs`: Chứa các tập trọng số sau khi huấn luyện 
- `config.py`: Chứa các tham số cấu hình cho dự án.
- `utils.py`: Xử lý việc tải dữ liệu, tiền xử lý và trực quan hóa, triển khai hàm mất mát Triplet Loss và tạo triplet.
- `model`: Định nghĩa kiến trúc mạng nơ-ron và quy trình huấn luyện.
- `inference.py`: huấn luyện và đánh giá mô hình
- `main.py`: Tập lệnh chính tích hợp toàn bộ quy trình.
- `app.py`: Tệp chính chứa mã nguồn API Flask.

## 3. Cài đặt

### Cài đặt các phụ thuộc

Đảm bảo rằng bạn đã cài đặt các thư viện cần thiết. Bạn có thể cài đặt chúng bằng cách sử dụng `pip`:

```bash
pip install numpy flask pillow scikit-learn joblib
```

## 4. Cách sử dụng
Khởi chạy ứng dụng Flask bằng cách chạy:
```bash
python app.py
```