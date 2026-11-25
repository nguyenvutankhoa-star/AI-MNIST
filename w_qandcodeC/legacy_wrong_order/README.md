# PTQ và QAT cho Golden Model C

Hai file C đã có chú thích tiếng Việt ở từng dòng thao tác, khai báo, vòng lặp, điều kiện và công thức tính chỉ số.

| Thành phần | Nội dung |
|---|---|
| `mnist_PTQ.c` | Suy luận INT8 với trọng số PTQ, không huấn luyện lại |
| `mnist_QAT.c` | Suy luận INT8 với trọng số đã tinh chỉnh QAT |
| `PTQ/`, `QAT/` | Hai bộ trọng số riêng biệt: TXT, BIN, header, scale và checkpoint float |
| `export_models.py` | Nạp trọng số Golden, gộp BN, calibration PTQ, huấn luyện QAT và xuất trọng số |
| `verify_models.py` | Biên dịch, kiểm tra C trên toàn bộ tập test, so sánh tham chiếu và kiểm tra ảnh lỗi |
| `training_report.json` | Cấu hình, lịch sử QAT, SHA256 nguồn trọng số và kết quả tham chiếu |
| `verification_report.json` | Kết quả kiểm chứng thực tế hai bản C và Golden C |
| `*_test_results.txt` | Log đánh giá C và logits của ảnh TXT mẫu |

## Biên dịch và chạy

Mở PowerShell tại `D:\reports`. Dấu ngoặc kép cần thiết vì tên thư mục chứa ký tự `&`.

```powershell
gcc -O3 -std=c11 -Wall -Wextra -Werror "w_q&codeC/mnist_PTQ.c" -lm -o "w_q&codeC/mnist_PTQ.exe"
gcc -O3 -std=c11 -Wall -Wextra -Werror "w_q&codeC/mnist_QAT.c" -lm -o "w_q&codeC/mnist_QAT.exe"

# Ảnh mặc định data/input_image.txt, giống Golden Model
& ".\w_q&codeC\mnist_PTQ.exe"
& ".\w_q&codeC\mnist_QAT.exe"

# Ảnh TXT khác: đúng 784 số thực [0,1], phân tách bởi khoảng trắng hoặc dấu phẩy
& ".\w_q&codeC\mnist_PTQ.exe" "data/input_image.txt"

# Toàn bộ 10000 ảnh test có sẵn, tùy chọn ghi nhãn dự đoán int32 little-endian
& ".\w_q&codeC\mnist_PTQ.exe" --test data "w_q&codeC/PTQ_c_predictions.bin"
& ".\w_q&codeC\mnist_QAT.exe" --test data "w_q&codeC/QAT_c_predictions.bin"
```

Cũng có thể chạy EXE ngay trong `w_q&codeC` mà không truyền tham số, hoặc với `--test`; chương trình tự tìm `../data`. Từ vị trí khác, hãy truyền đường dẫn ảnh hoặc thư mục dữ liệu đầy đủ. Suy luận C không cần Python/TensorFlow. Cần giữ thư mục trọng số tương ứng bên cạnh file C khi **biên dịch** vì C nhúng mảng từ `PTQ/weights.h` hoặc `QAT/weights.h`. EXE đã biên dịch không cần mở các file trọng số.

## Kết quả đã chạy

| Mô hình | Ảnh đúng / 10000 | Độ chính xác |
|---|---:|---:|
| Golden C float gốc | 9037 | 90,37% |
| PTQ C INT8 | 9051 | 90,51% |
| QAT C INT8 | 9879 | 98,79% |

Hai bản C đều khớp **10000/10000 nhãn** với tham chiếu fake quantization TensorFlow tương ứng. Golden C gốc cũng khớp toàn bộ nhãn với tham chiếu float đã gộp BN. Bộ kiểm tra đã chạy với GCC `-Wall -Wextra -Werror`, kiểm tra ảnh TXT thiếu/thừa pixel và NaN. Ảnh mẫu được cả hai bản dự đoán là số 7.

**Giải thích baseline:** yêu cầu lấy `Source C/Golden Model AI.c` làm chuẩn, nên kiến trúc được giữ là `Conv -> BN -> ReLU -> Pool`, hai lần, rồi `Dense`. Mã Python hiện có trong dự án mô tả `Conv -> ReLU -> BN -> Pool`; hai thứ tự này không tương đương. Các kết quả trên là của đúng **Golden C được cung cấp**, không lấy số accuracy từ notebook hay mô hình Keras. Chênh lệch của QAT có đóng góp từ quá trình tinh chỉnh có nhãn trên kiến trúc này; không thể quy toàn bộ mức tăng cho riêng lượng tử hóa.

## PTQ được thực hiện như thế nào?

1. Đọc đúng các tên TXT mà Golden C sử dụng trong `weights/`; không dùng các file trùng tên có khoảng trắng đầu tên.
2. Gộp BatchNorm với epsilon `0.001`: `factor = gamma/sqrt(var+eps)`, `W_fold = W*factor`, `b_fold = (b-mean)*factor+beta`.
3. Chọn 1024 ảnh từ phần train để calibration hai activation sau ReLU. Không dùng ảnh test để chọn scale.
4. Dùng INT8 đối xứng theo từng tensor: `scale = 2^ceil(log2(max_abs/127))`, `q = clip(round_away(x/scale), -127, 127)`. Input dùng max thực bằng 1; ReLU output không âm.
5. Bias dùng INT32 với scale bằng `input_scale * weight_scale`. Toàn bộ bộ xuất đã kiểm tra cận tích lũy INT32 của từng lớp.

Scale là đơn vị thực của một bước số nguyên, không phải hệ số nhân nghịch đảo. Vì scale là lũy thừa hai, giá trị 1 của ảnh ứng với số nguyên 64 ở scale đầu vào `2^-6`. Miền biểu diễn vẫn là [-127,127]; không nhất thiết dùng hết các mức của miền INT8.

## QAT được thực hiện như thế nào?

`export_models.py` thực hiện QAT tùy chỉnh bằng TensorFlow `GradientTape` và straight-through estimator (STE); đây không phải thao tác đổi tên bộ trọng số PTQ, và không sử dụng converter TFLite.

- Khởi tạo từ chính trọng số float đã gộp BN của Golden C.
- Dùng seed 2026 để chia 60000 ảnh MNIST train thành 55000 ảnh tinh chỉnh và 5000 ảnh validation.
- Giữ cố định các scale PTQ để phù hợp kernel C; mô phỏng lượng tử hóa input, weights, bias và hai activation trong mọi lượt forward.
- Forward của fake quantization làm tròn giống C; backward truyền gradient STE trong miền biểu diễn, gradient ngoài miền bị chặn.
- Cập nhật cả weights và bias bằng Adam, learning rate `1e-4`, batch size 128, 3 epoch.
- Chọn checkpoint có accuracy validation tốt nhất: epoch 2, validation 98,80%. Sau đó xuất INT8/INT32.
- 10000 ảnh test chỉ được dùng sau khi chọn checkpoint; không tham gia calibration, cập nhật gradient hoặc chọn epoch.

Hai file C có kernel suy luận giống nhau là đúng: QAT là sự khác biệt trong quá trình huấn luyện và bộ trọng số cuối cùng. Chi tiết QAT nói chung có thể xem tại [tài liệu TensorFlow](https://www.tensorflow.org/model_optimization/guide/quantization/training); cấu hình ở đây là một triển khai tùy chỉnh cho kernel C này.

## Scale và định dạng trọng số

Mỗi thư mục `PTQ`/`QAT` chứa:

- `layer1_w`, `layer2_w`, `layer3_w`: INT8, lần lượt 72, 1152, 4000 phần tử.
- `layer1_b`, `layer2_b`, `layer3_b`: INT32, lần lượt 8, 16, 10 phần tử.
- Mỗi tensor có `.txt` (một số nguyên mỗi dòng) và `.bin` (raw little-endian, không header).
- `weights.h`: cùng các giá trị, nhúng sẵn cho C; đây là nguồn trọng số C thực sự dùng khi biên dịch.
- `scales.json`: shape, số mũ scale đầu vào/trọng số/đầu ra/bias và quy tắc làm tròn.
- `float_checkpoint.npz`: các tensor float đã gộp BN để tái kiểm tra hoặc tinh chỉnh tiếp; không phải file C đọc.

| Lớp | Layout weights | Shape | Input exp | Weight exp | Output exp | Bias exp |
|---|---|---|---:|---:|---:|---:|
| Conv1 | HWIO | 3×3×1×8 | -6 | -3 | -3 | -9 |
| Conv2 | HWIO | 3×3×8×16 | -3 | -7 | -3 | -10 |
| Dense | IO | 400×10 | -3 | -8 | -11 | -11 |

Không có trọng số BN riêng sau khi gộp. Chuyển scale sau Conv theo `shift = input_exp + weight_exp - output_exp`. C dùng INT64 ở bước nhân/chia đổi scale và làm tròn; các phép MAC dùng INT32. ReLU, requantization và max pooling được hợp nhất để không cần lưu cả feature map Conv. Logits Dense giữ INT32, argmax trực tiếp; chỉ giải lượng tử bằng float khi in kết quả một ảnh. Không dùng Softmax.

Tổng dữ liệu weights+bias lượng tử hóa là `5224 + 34*4 = 5360 byte`, chưa tính scale, mã máy và định dạng lưu trữ. TXT/header đương nhiên có kích thước lớn hơn BIN.

## Tái tạo và kiểm chứng

```powershell
# Cần môi trường Python có NumPy và TensorFlow (lần tạo này dùng TF 2.21.0)
& ".\.venv\Scripts\python.exe" "w_q&codeC/export_models.py" --epochs 3

# Có thể chỉ rõ file MNIST cục bộ khác
& ".\.venv\Scripts\python.exe" "w_q&codeC/export_models.py" --mnist "C:/Users/ADMIN/.keras/datasets/mnist.npz" --epochs 3

# Biên dịch lại EXE và đối chiếu cả 10000 ảnh với các reference predictions vừa xuất
& ".\.venv\Scripts\python.exe" "w_q&codeC/verify_models.py"
```

Xuất lại sẽ ghi đè bộ trọng số và tham chiếu trong thư mục này; sau đó phải biên dịch lại C để EXE dùng trọng số mới. Seed, split, phiên bản TensorFlow, SHA256 của dữ liệu MNIST và các TXT nguồn được lưu trong báo cáo; phép toán float có thể có sai khác nhỏ giữa môi trường máy khác nhau. Không sửa các file nguồn C hoặc trọng số gốc bên ngoài thư mục này.
