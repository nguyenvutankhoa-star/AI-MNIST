# TỔNG HỢP KIẾN THỨC MẠNG NƠ-RON TÍCH CHẬP (CNN)

## 1. Chuẩn bị và Tiền xử lý Dữ liệu (Load Data)
* **Khởi tạo dữ liệu**: Sử dụng tập dữ liệu MNIST gồm các ảnh chữ số viết tay từ 0 đến 9, kích thước mỗi ảnh ban đầu là 28x28 pixel.
* **Tiền xử lý (Preprocessing)**: Ép kiểu và thêm chiều không gian (kênh màu) cho ảnh xám thành `(28, 28, 1)`. Sau đó, chuẩn hóa các giá trị điểm ảnh từ 0-255 về khoảng `0.0` đến `1.0` bằng phép chia `/ 255.0` để mô hình dễ hội tụ.
* **Mã hóa One-hot (One-hot Encoding)**: Chuyển đổi nhãn phân loại từ dạng số nguyên thành mảng xác suất (ví dụ: nhãn số 3 biến thành mảng `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`).

## 2. Kiến trúc Mô hình và Công thức (Build Model)
* **Lớp Conv2D (Tích chập)**: Sử dụng các bộ lọc (kernel) trượt qua ảnh để trích xuất các đặc trưng cơ bản (nét ngang, dọc, cong).
  * *Kích thước đầu ra*: O = I - K + 1 (với I là kích thước ảnh, K là kích thước bộ lọc, stride bằng 1).
  * *Số tham số*: P = (K_h x K_w x C_in + 1) x C_out (với C_in là số kênh đầu vào, C_out là số lượng bộ lọc).
* **Lớp BatchNormalization**: Chuẩn hóa dữ liệu đầu ra của lớp trước đó để mạng nơ-ron học ổn định và nhanh hơn.
  * *Số tham số*: P = 4 x C_in (bao gồm gamma, beta, moving mean, moving variance).
* **Lớp MaxPooling2D**: Lọc lấy điểm ảnh có giá trị lớn nhất trong từng vùng nhỏ, giúp giảm kích thước không gian ảnh đi một nửa, giữ lại đặc trưng cốt lõi và loại bỏ nhiễu.
  * *Số tham số*: P = 0.
* **Lớp Flatten và Dropout**: Lớp Flatten có nhiệm vụ duỗi thẳng khối dữ liệu 3 chiều thành 1 mảng 1 chiều. Lớp Dropout vô hiệu hóa ngẫu nhiên một phần nơ-ron (ví dụ 25%) để chống hiện tượng học vẹt (Overfitting). Cả hai lớp này đều có số tham số P = 0.
* **Lớp Dense (Fully Connected)**: Lớp phân loại cuối cùng, tính điểm số cho từng nhãn dựa trên tất cả các đặc trưng đã trích xuất, sau đó dùng hàm kích hoạt Softmax để xuất ra xác suất.
  * *Công thức tính điểm*: z = (w1*x1 + w2*x2 + ... + wn*xn) + bias
  * *Số tham số*: P = (N_in + 1) x N_out (với N_in là số đặc trưng đầu vào, N_out là số nơ-ron đầu ra).

## 3. Biên dịch và Huấn luyện (Compile & Train)
* **Biên dịch (Compile)**: Cài đặt thuật toán tối ưu `Adam` để tự động điều chỉnh trọng số, hàm mất mát `categorical_crossentropy` để đo lường mức độ sai lệch, và chỉ số `accuracy` để theo dõi độ chính xác.
* **Huấn luyện (Fit)**: Mô hình duyệt qua toàn bộ dữ liệu nhiều vòng (`epochs`). Ở mỗi vòng, dữ liệu được chia nhỏ thành từng lô (`batch_size = 32`) để mô hình tính toán, sửa sai và cập nhật trọng số liên tục.
