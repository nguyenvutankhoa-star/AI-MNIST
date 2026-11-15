# Kết quả PTQ và QAT

## Hai tệp C độc lập

- `D:\reports\Source C\mnist_PTQ.c`: lượng hóa sau huấn luyện, dùng trọng số float đã lưu trong `weights`.
- `D:\reports\Source C\mnist_QAT.c`: khởi tạo từ cùng trọng số, tinh chỉnh cả hai Conv và Dense bằng fake quantization INT8, clipped straight-through estimator (STE) và SGD; sau đó đánh giá bằng suy luận số nguyên.

Không thay đổi mã nguồn cũ hoặc trọng số gốc. Các bản tham số có khoảng trắng đầu tên trong `weights` đã được đối chiếu: nội dung trùng với tên chuẩn. Mã mới đọc tên chuẩn, kiểm tra số phần tử, giá trị hữu hạn, kích thước dữ liệu và nhãn.

## Dữ liệu và thiết lập

Bộ `C:\Users\ADMIN\.keras\datasets\mnist.npz` có sẵn trên máy cung cấp 60.000 ảnh train. Đã kiểm tra 10.000 nhãn và toàn bộ pixel test trong cache khớp chính xác với `data/test_images.bin`, `data/test_labels.bin` của dự án. Chỉ train được dùng để hiệu chuẩn và huấn luyện; test chỉ dùng đánh giá.

- Ảnh: float32 little-endian, 28×28, giá trị [0,1]; nhãn int32 little-endian, 0–9.
- Kiến trúc: valid Conv 3×3×1×8 → ReLU → BN → Pool2 → valid Conv 3×3×8×16 → ReLU → BN → Pool2 → Flatten400 → Dense10.
- Hiệu chuẩn activation: cực đại trị tuyệt đối từ 512 ảnh train đầu tiên. Trọng số: lượng hóa đối xứng theo từng tensor, khoảng [-127,127]. Các scale đều là lũy thừa của 2, dùng chung cho đầu ra Dense để so sánh logits số nguyên trực tiếp.
- PTQ: không cập nhật trọng số.
- QAT: 1 epoch, 60.000 ảnh theo thứ tự cố định, SGD từng ảnh, learning rate 0,0001; BN và scale đóng băng; bỏ Dropout khi tinh chỉnh. Đây là cấu hình thử nghiệm cụ thể, chưa tìm siêu tham số tối ưu.
- QAT giữ trọng số float để cập nhật, fake quant trọng số và activation khi forward, lan truyền gradient qua STE. Bias giữ float trong huấn luyện, chuyển INT32 khi xuất. Vì vậy forward huấn luyện là mô phỏng lượng hóa; kết quả báo cáo bên dưới lấy từ kernel integer triển khai thực tế.

## Kết quả kiểm thử thực tế

| Chỉ tiêu | PTQ | QAT |
|---|---:|---:|
| Số ảnh test | 10.000 | 10.000 |
| Dự đoán INT8 đúng | 9.874 | 9.888 |
| Độ chính xác INT8 | 98,74% | 98,88% |
| Dự đoán sai | 126 | 112 |
| Độ chính xác float cùng trạng thái trọng số | 98,69% | 98,87% |
| Đồng thuận INT8 với float | 99,84% | 99,83% |
| Loss train trung bình, epoch cuối | không huấn luyện | 0,030349 |
| Trọng số INT8 | 5.224 byte | 5.224 byte |
| Tệp trọng số FPGA, gồm metadata và bias | 5.736 byte | 5.736 byte |
| Buffer activation integer, gồm input | 2.536 byte | 2.536 byte |

QAT thêm 14 dự đoán đúng, tăng 0,14 điểm phần trăm và giảm khoảng 11,11% số lỗi so với PTQ. Không có bảo đảm QAT luôn tốt hơn trên dữ liệu khác.

Log đo CPU: PTQ suy luận integer 2,583 giây/10.000 ảnh; QAT 5,664 giây/10.000 ảnh; QAT huấn luyện 542,758 giây. Có tác vụ chạy đồng thời trong phiên đo nên **không dùng các số thời gian này để kết luận tốc độ PTQ/QAT**. Hai bản có cùng kernel suy luận integer và cùng kích thước mô hình; chưa tổng hợp FPGA, chưa đo LUT/DSP/BRAM, timing hoặc công suất.

## Gộp BatchNorm đúng thứ tự gốc

Mô hình Keras lưu `Conv(activation=relu) → BN`, không phải `Conv → BN → ReLU`. Không được đổi thứ tự này khi gộp.

Với từng kênh, đặt `a = gamma / sqrt(var + 0.001)` và `t = beta - a*mean`:

```
W_fold = a * W
b_fold = a * b + t
z = Conv(x, W_fold, b_fold)
y = max(z, t) nếu a >= 0; ngược lại y = min(z, t)
```

Như vậy BN được gộp vào trọng số và bias; ngưỡng kích hoạt đã biến đổi giữ đúng ReLU trước BN, kể cả gamma âm. Không có phép căn, chia, float hay lớp BN riêng trong `conv_int`/`infer_int`. Conv, ngưỡng kích hoạt và Pool được tính trong cùng vòng lặp; không tạo feature map Conv đầy đủ. Kênh cuối/hàng cuối bị Pool valid bỏ qua cũng không phải tính.

MAC dùng INT32, tích trung gian requantization dùng INT64, làm tròn đối xứng rồi bão hòa INT8. Scale lũy thừa 2 cho phép hiện thực phép chia bằng dịch bit và logic làm tròn trên FPGA. INT64 là kiểu C trung gian an toàn; cần thu hẹp độ rộng theo ràng buộc thực tế khi chuyển HLS/RTL. Phần đọc tệp, hiệu chuẩn và huấn luyện chạy trên host, không đưa toàn bộ chương trình C có file I/O lên FPGA. Buffer và dung lượng trong bảng chỉ là dữ liệu suy luận, không bao gồm cấu trúc float của chương trình host, stack, alignment hoặc metadata.

## Xác minh độc lập

- GCC C11: cả hai tệp biên dịch thành công với `-Wall -Wextra -Wpedantic -Werror`.
- NumPy tính nguyên bản Conv → ReLU → BN đối chiếu phiên bản folded trên toàn bộ 10.000 ảnh: sai số logits tối đa `8.58306885e-06`; trong dung sai `atol=3e-5, rtol=1e-5`.
- Nhãn float của C khớp mô hình NumPy nguyên bản **10.000/10.000 ảnh**.
- Đọc lại từng tệp trọng số FPGA bằng NumPy, tính convolution integer độc lập: nhãn khớp C **256/256 ảnh cho PTQ và 256/256 ảnh cho QAT**.
- Đã thử biên dịch UndefinedBehaviorSanitizer nhưng GCC hiện tại thiếu thư viện `libubsan`; không có kết quả sanitizer và không coi bước này là đã đạt.

Script đối chiếu: `artifacts/verify_models.py`. Log: `PTQ_results.txt`, `QAT_results.txt`, `verification_results.txt` trong cùng thư mục.

## Chạy lại

Từ PowerShell tại `D:\reports`:

```powershell
# Chỉ cần khi muốn xuất lại dữ liệu train từ cache; cần NumPy.
.venv/Scripts/python.exe artifacts/prepare_train.py

gcc -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror "Source C/mnist_PTQ.c" -lm -o artifacts/ptq.exe
gcc -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror "Source C/mnist_QAT.c" -lm -o artifacts/qat.exe

artifacts/ptq.exe 0 60000 10000
artifacts/qat.exe 1 60000 10000
.venv/Scripts/python.exe artifacts/verify_models.py
```

Tham số lần lượt là số epoch, số ảnh train sử dụng, số ảnh test. PTQ yêu cầu epoch bằng 0. Cần chạy từ thư mục gốc để đọc đúng đường dẫn tương đối. Mỗi lần QAT chạy đều bắt đầu từ trọng số gốc trong `weights`; chương trình chưa hỗ trợ resume từ checkpoint. Chạy lại sẽ ghi đè tệp export và predictions tương ứng.

## Định dạng tệp xuất

`artifacts/PTQ_fpga.bin`, `artifacts/QAT_fpga.bin` lưu little-endian trên máy đã kiểm thử:

1. 4 byte magic ASCII `MNI8`, một INT32 version=1.
2. Lặp theo thứ tự Conv1, Conv2, Dense: tám INT32 metadata `[in, ci, co, pool_side, weight_count, input_exponent, weight_exponent, output_exponent]`.
3. `weight_count` trọng số INT8, theo thứ tự HWIO cho Conv và `[input,output]` cho Dense.
4. `co` bias INT32, `co` threshold INT32 trong cùng đơn vị accumulator; tiếp theo `co` dấu INT32 (+1/-1 cho Conv). Threshold và dấu của Dense không dùng.

Giá trị thực xấp xỉ `q * 2^exponent`; bias được chia cho `2^(input_exponent+weight_exponent)` rồi làm tròn. Với Conv: chặn accumulator theo threshold và dấu trước requantization; exponent chuyển đổi bằng `input_exponent+weight_exponent-output_exponent`. Dense xuất INT32 và argmax, không cần softmax. Tệp predictions gồm hai INT32 mỗi ảnh `[nhãn float, nhãn integer]`.

`weights_quantized` cũ không được dùng: script cũ xuất hệ số BN bằng hằng 128/2^7=1, không biểu diễn BN đã học, không truyền scale activation giữa lớp và có dấu hiệu dữ liệu INT8 30×30 không khớp ảnh float 28×28. Mã mới dùng trực tiếp nguồn float và dữ liệu float đã kiểm chứng.
