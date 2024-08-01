# MNIST Dataset

You can download the MNIST dataset from OpenML using the following link:

[Download MNIST Dataset](https://www.openml.org/search?type=data&sort=runs&id=554&status=active)

This dataset is used for training and testing.



# Logistic Regression trên tập dữ liệu MNIST với Numpy
## 1. Logistic Regression
Hồi quy logistic là một thuật toán học máy có giám sát nhằm hoàn thành các nhiệm vụ phân loại nhị phân bằng cách dự đoán xác suất của một kết quả, sự kiện hoặc quan sát. Mô hình đưa ra kết quả nhị phân hoặc phân đôi được giới hạn ở hai kết quả có thể xảy ra: có/không, 0/1 hoặc đúng/sai

## 2. Logistic Regression – Sigmoid Function
<image src = "Logistic_Regression_formula.png">

- $x$ = giá trị đầu vào
- $y$ = predict output
- $b0$ = bias or intercept term
- $b1$ = hệ số đầu vào $x$

<image src = "graphical_representation.png">


## 3. Cách tiếp cận bài toán trên

### 1. Xây dựng mô hình phân loại đa lớp

- **Mục tiêu**: Phân loại các hình ảnh chữ số (0-9) từ bộ dữ liệu MNIST thành các lớp tương ứng.

### 2. Chọn mô hình

- **Logistic Regression**: Sử dụng làm mô hình cơ bản để phân loại nhị phân.
- **One-vs-All**: Áp dụng để giải quyết vấn đề phân loại đa lớp, trong đó một mô hình Logistic Regression riêng biệt được đào tạo cho mỗi lớp.

### 3. Tiền xử lý dữ liệu

- **Chuẩn hóa dữ liệu**: Chuyển đổi hình ảnh từ bộ dữ liệu MNIST thành các vector và chuẩn hóa để sử dụng trong mô hình.

### 4. Huấn luyện mô hình

- **Đào tạo từng mô hình cho mỗi lớp**: 
  - Đối với mỗi lớp (0-9), tạo ra một nhãn nhị phân (0 hoặc 1) để phân biệt lớp đó với các lớp khác.
  - Huấn luyện một mô hình Logistic Regression cho từng lớp.

- **Gradient Descent**: 
  - Sử dụng thuật toán Gradient Descent để tối ưu hóa trọng số của mỗi mô hình, nhằm giảm thiểu hàm mất mát (cost function).

### 5. Dự đoán

- **Kết hợp kết quả từ các mô hình**: 
  - Sau khi tất cả các mô hình được đào tạo, dự đoán xác suất cho từng lớp đối với một mẫu mới.

- **Chọn lớp có xác suất cao nhất**: 
  - Lớp với xác suất cao nhất từ các mô hình được chọn làm kết quả cuối cùng.



## 4. Files
- `main.py`: Chứa một pipeline đơn giản sử dụng Logistic Regression trên tập dữ liệu MNIST.
- `model.py`: Triển khai Logistic Regression sử dụng Numpy.
- `utils.py`: Load MNIST dataset
- `notebook/Random_Forest_with_MNIST_Dataset.ipynb`: Ví dụ triển khai Logistic Regression trên tập dữ liệu MNIST cho EXERCISES 1.