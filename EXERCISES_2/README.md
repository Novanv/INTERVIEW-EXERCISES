# Triplet Loss

## 1. FILES

- `simple_triplet_loss.py`:  Chứa triển khai Triplet Loss với 1 mẫu neo và 1 mẫu giả
- `extend_triplet_loss.py`: Chứa triển khai Triplet Loss với 2 mẫu neo và 5 mẫu giả


## 2. Định nghĩa
Triplet Loss là một hàm mất mát thường được sử dụng trong các vấn đề Image Regconition và Matching Problems, đặc biệt là trong các mô hình Deep LEarning. Mục tiêu của Triplet Loss là đảm bảo rằng các mẫu thuộc cùng một lớp gần nhau hơn trong không gian đặc trưng so với các mẫu thuộc các lớp khác nhau.


## 3. Triplet Loss Explanation


$$\text{Loss} = \sum_{i=1}^{N} \left[ \left\| f_i^a - f_i^p \right\|_2^2 - \left\| f_i^a - f_i^n \right\|_2^2 + \alpha \right]$$
Trong đó:
- $( f_i^a )$: Đặc trưng của mẫu ảnh chính (anchor)
- $( f_i^p )$: Đặc trưng của mẫu ảnh dương (positive)
- $( f_i^n )$: Đặc trưng của mẫu ảnh âm (negative)
- $( \alpha )$: là margin
- $( \left\| \cdot \right\|_2 )$: Norm 2 (hay Euclidean distance)

## 4. Extended Triplet Loss Explanation

$$
\mathcal{L}(a, P, N) = \max \left( 0, \frac{1}{|P|} \sum_{p \in P} \| f(a) - f(p) \|_2^2 - \frac{1}{|N|} \sum_{n \in N} \| f(a) - f(n) \|_2^2 + \alpha \right)
$$

Trong đó:
- $f_a$ là anchor.
- $f_p$ là positive sample.
- $f_n$ là negative sample.
- $P$ là số lượng của positive samples.
- $N$ là số lượng của negative samples.
- $α$ là margin.
- $( \left\| \cdot \right\|_2 )$: Norm 2 (hay Euclidean distance)

## 5. Giải thích công thức
### 5.1. Hàm Embedding $f$
- $f(x)$: Một hàm (thường là mạng nơ-ron) ánh xạ một đầu vào $x$ đến một không gian embedding nơi có thể thực hiện các so sánh.
### 5.2. Distance Metric
- $\|f(x_i^a) - f(x_i^p)\|^2$: Khoảng cách bình phương giữa anchor và positive sample trong không gian embedding. (L2 Distance)

- $\|f(x_i^a) - f(x_i^n)\|^2$: Khoảng cách bình phương giữa anchor và negative sample trong không gian embedding. (L2 Distance)
### 5.3. Margin $\alpha $
- Margin được áp dụng giữa các cặp positive và negative. Margin giúp đảm bảo negative samples cách xa the anchor hơn positive examples ít nhất $\alpha$.

## 6. Ưu, nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-------------|
| Tăng khả năng phân biệt: Giúp mô hình học các đặc trưng phân biệt, làm tăng khả năng nhận diện chính xác.| Khó khăn trong việc chọn triplets: Việc chọn ra các triplet tốt (anchor, positive, negative) là một thách thức. |
| Không yêu cầu nhãn chi tiết: Chỉ cần biết cặp nào cùng lớp và cặp nào khác lớp. | Cần nhiều thời gian huấn luyện: Tính toán khoảng cách giữa nhiều cặp đối tượng trong mỗi batch làm tăng thời gian huấn luyện. |
| Hiệu quả trong việc học nhúng: Tạo ra các nhúng vector giúp tính toán khoảng cách giữa các đối tượng dễ dàng. | Độ phức tạp trong triển khai: Quản lý và lựa chọn các triplet hiệu quả có thể phức tạp. |
| Khả năng tổng quát hóa tốt: Tốt khi xử lý dữ liệu chưa từng gặp. | Nhạy cảm với tham số biên (margin): Kết quả có thể nhạy cảm với giá trị của tham số biên, yêu cầu lựa chọn cẩn thận. |

## 7. Ứng dụng

1. **Nhận dạng khuôn mặt:** Sử dụng Triplet Loss để đảm bảo khuôn mặt của cùng một người gần nhau hơn trong không gian nhúng so với khuôn mặt của người khác.
2. **Nhận diện chữ viết tay:** Dùng Triplet Loss để phân biệt giữa các chữ viết tay của các cá nhân khác nhau.
3. **Tìm kiếm hình ảnh:** Triplet Loss giúp tìm kiếm hình ảnh tương tự trong một tập dữ liệu lớn.
4. **Phân loại ảnh y tế:** Sử dụng Triplet Loss để phân loại các loại tế bào ung thư khác nhau.
5. **Nhận dạng giọng nói:** Áp dụng Triplet Loss để phân biệt giọng nói của các cá nhân khác nhau trong các hệ thống nhận dạng giọng nói.