# PTQ/QAT V2 — đã sửa thứ tự ReLU và BatchNorm

Hai file C hiện tại khớp kiến trúc checkpoint Keras: **Conv → ReLU → BatchNorm → Pool**, lặp hai lần rồi Flatten/Dense. Chú thích tiếng Việt giải thích từng dòng thao tác. Khi chạy, dòng đầu có `V2: Conv-ReLU-BN` để nhận biết EXE mới.

## Chạy trên toàn bộ 10000 ảnh

Mở PowerShell tại `D:\reports`:

```powershell
# Không tham số cũng tự kiểm tra đủ 10000 ảnh
& ".\w_qandcodeC\mnist_PTQ.exe"
& ".\w_qandcodeC\mnist_QAT.exe"

# Cách chạy tường minh và tùy chọn lưu nhãn dự đoán int32 little-endian
& ".\w_qandcodeC\mnist_PTQ.exe" --test data "w_qandcodeC/PTQ_c_predictions.bin"
& ".\w_qandcodeC\mnist_QAT.exe" --test data "w_qandcodeC/QAT_c_predictions.bin"

# Muốn kiểm tra riêng một ảnh thì truyền đường dẫn TXT: 784 pixel float [0,1]
& ".\w_qandcodeC\mnist_PTQ.exe" "data/input_image.txt"
```

Có thể chạy từ `w_qandcodeC` hoặc `Source C`: chương trình tự tìm `../data`. Từ thư mục khác, dùng `--test` và đường dẫn đầy đủ đến thư mục dữ liệu. Chương trình báo tiến độ mỗi 1000 ảnh.

## Kết quả V2 đã chạy

| Mô hình | Đúng / 10000 | Accuracy |
|---|---:|---:|
| Keras float/TXT float/Golden C đã sửa | 9869 | 98,69% |
| PTQ C V2 | 9874 | 98,74% |
| QAT C V2 | 9896 | 98,96% |

Hai bản C khớp **10000/10000 nhãn** với tham chiếu fake-quant TensorFlow tương ứng. PTQ không huấn luyện lại. QAT dùng 55000 ảnh train, 5000 ảnh validation, seed 2026, Adam `1e-4`, batch 128, 3 epoch; chọn epoch 3 theo validation. Test không dùng để calibration hoặc chọn epoch. Accuracy PTQ nhỉnh hơn float ở lần đánh giá này là kết quả quan sát trên tập test, không phải bảo đảm lượng tử hóa luôn tăng accuracy.

Đã kiểm tra biên dịch GCC với `-Wall -Wextra -Werror`, chạy mặc định đủ 10000 ảnh, xử lý ảnh TXT thiếu/thừa/NaN, phép làm tròn âm và hướng chặn khi gamma âm. Kết quả chi tiết: `verification_report.json`, `training_report.json`, `PTQ_test_results.txt`, `QAT_test_results.txt`.

## Vì sao bản trước đạt khoảng 90%?

Golden trước đây chạy `Conv → BN → ReLU`, khác với checkpoint dùng ReLU bên trong Conv rồi mới BN. Bản PTQ trước đã kế thừa Golden đó. Bộ kiểm toán đã cô lập việc đổi thứ tự làm baseline float giảm từ 98,69% xuống 90,37%. V2 đã sửa Golden C, notebook Inference và bộ xuất/suy luận lượng tử. Bản cũ được lưu riêng trong `legacy_wrong_order/`; không chạy EXE ở thư mục lưu trữ này để đánh giá V2.

`audit/BAO_CAO_KIEM_TRA.md` và kết quả kiểm toán cũ mô tả trạng thái **trước V2**, được giữ làm bằng chứng nguyên nhân. Mã notebook training vẫn cần thống nhất cấu hình 30×30 hiện có với checkpoint 28×28 nếu muốn huấn luyện lại từ đầu; V2 sử dụng checkpoint/trọng số TXT 28×28 đang có và không lấy output accuracy cũ của notebook làm baseline.

## Cách giữ đúng Conv → ReLU → BN bằng số nguyên

Với `z=Conv(x)`, `alpha=gamma/sqrt(var+0.001)` và `t=beta-alpha*mean`:

```text
BN(ReLU(z)) = alpha*max(z,0)+t
             = max(alpha*z+t,t) nếu alpha >= 0
             = min(alpha*z+t,t) nếu alpha < 0
```

V2 biến đổi weights và bias thành `W'=alpha*W`, `b'=alpha*b+t`, đồng thời **bắt buộc xuất ngưỡng t=BN(0) và hướng chặn d=sign(alpha)**. Không áp ReLU tại 0 sau BN. Trường hợp alpha=0 được biểu diễn bằng đầu ra hằng t. Mọi giá trị âm sau BN được giữ đúng; pooling khởi tạo -127.

INT8 dùng zero-point 0, miền [-127,127], scale lũy thừa hai: `x_real=q*2^exponent`. Mỗi tensor weight và mỗi activation có scale riêng. Làm tròn gần nhất, hòa ra xa 0; bias và threshold dùng INT32 với scale `input_scale*weight_scale`. MAC dùng INT32, bước đổi scale dùng INT64. Bộ xuất kiểm tra cận tràn trước khi lưu.

Calibration dùng 1024 ảnh từ phần train; hai output sau BN được đo cả giá trị âm và dương. QAT mô phỏng lượng tử weights, bias, ngưỡng BN(0) và activation trên cùng kernel biến đổi; gradients dùng STE. Ngưỡng BN và hướng chặn giữ cố định trong quá trình tinh chỉnh. Đây là QAT tùy chỉnh, không phải converter TFLite.

## Trọng số mới

`PTQ/` và `QAT/` chứa:

- `layer1_w`, `layer2_w`, `layer3_w`: weights INT8, shape HWIO `(3,3,1,8)`, `(3,3,8,16)`, IO `(400,10)`.
- `layer1_b`, `layer2_b`, `layer3_b`: bias INT32, 8/16/10 phần tử.
- `layer1_t`, `layer2_t`: ngưỡng BN(0) INT32 trong scale accumulator, 8/16 phần tử.
- `layer1_d`, `layer2_d`: hướng chặn INT32, +1 dùng max, -1 dùng min, 8/16 phần tử.
- Mỗi tensor có TXT (mỗi dòng một số) và BIN (raw little-endian).
- `weights.h`: toàn bộ tham số nhúng cho C; **C thực sự dùng header này khi biên dịch**, không mở TXT/BIN lúc chạy.
- `scales.json`: format version 2, số mũ scale, quy tắc ngưỡng và shape.
- `float_checkpoint.npz`: tham số float đã biến đổi và ngưỡng/hướng chặn, dùng kiểm tra hoặc tinh chỉnh tiếp.

Tổng dữ liệu trọng số/bias/ngưỡng/hướng chặn raw là 5552 byte, không tính metadata, mã máy hoặc overhead TXT/header. Không bỏ các tensor t/d khi chuyển sang triển khai khác.

## Biên dịch và tái tạo

```powershell
gcc -O3 -std=c11 -Wall -Wextra -Werror "w_qandcodeC/mnist_PTQ.c" -lm -o "w_qandcodeC/mnist_PTQ.exe"
gcc -O3 -std=c11 -Wall -Wextra -Werror "w_qandcodeC/mnist_QAT.c" -lm -o "w_qandcodeC/mnist_QAT.exe"

# Cần NumPy/TensorFlow; lần tạo này dùng TensorFlow 2.21.0, dữ liệu MNIST đã lưu cục bộ
& ".\.venv\Scripts\python.exe" "w_qandcodeC/export_models.py" --epochs 3

# Sau xuất lại trọng số: biên dịch lại và kiểm tra cả 10000 ảnh
& ".\.venv\Scripts\python.exe" "w_qandcodeC/verify_models.py"
```

EXE không cần TensorFlow để suy luận. Các bản `Source C/mnist_PTQ.c` và `Source C/mnist_QAT.c` cũng đã được cập nhật V2 và cùng dùng trọng số trong `w_qandcodeC`; các EXE tương ứng đã biên dịch lại. Nguồn chính để chỉnh tiếp là hai file C trong `w_qandcodeC`. Thay header hoặc xuất trọng số mới luôn cần biên dịch lại EXE.
