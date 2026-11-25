# Kiểm tra luồng training → xuất trọng số → Python/C → PTQ/QAT

> Báo cáo lịch sử trước bản sửa V2. Hiện PTQ/QAT đã được sửa và tái xuất; xem `../README.md` và `../verification_report.json` để biết kết quả hiện tại. Các số 90% dưới đây được giữ để giải thích nguyên nhân, không phải accuracy của EXE V2.

Kiểm tra ngày 15/11/2026. Các phép đo dưới đây chạy trên toàn bộ 10000 ảnh test có sẵn. Script tái chạy: `w_qandcodeC/audit_pipeline.py`. Chi tiết máy đọc được: `audit_results.json`.

## 1. Nguyên nhân chính đã xác nhận

Mô hình đang lưu trong `mnist_cnn_model.keras` có hai khối **Conv có activation ReLU bên trong → BatchNorm → Pool**. Golden C và notebook Inference lại thực hiện **Conv tuyến tính → BatchNorm → ReLU → Pool**.

Đây là thay đổi hàm số của mạng, không chỉ là cách sắp xếp code:

```text
Đúng theo Keras: y = gamma * (max(0, Conv(x)) - mean) / sqrt(var + eps) + beta
Golden hiện có: y = max(0, gamma * (Conv(x) - mean) / sqrt(var + eps) + beta)
```

BatchNorm sau ReLU vẫn có thể tạo giá trị âm. Golden đưa ReLU xuống sau BatchNorm sẽ xóa các giá trị đó và thay đổi đặc trưng đi vào Conv2/Dense.

**Thử nghiệm cô lập:** giữ nguyên tất cả trọng số TXT, ảnh, epsilon, kernel, layout và pooling, chỉ đổi thứ tự BN/ReLU ở hai khối:

| Luồng kiểm tra | Đúng / 10000 | Accuracy |
|---|---:|---:|
| Keras đang lưu, `training=False` | 9869 | 98,69% |
| TXT với đúng thứ tự ReLU → BN | 9869 | 98,69% |
| `notebooks/infer_python.py` hiện tại | 9869 | 98,69% |
| C `Source C/Time.c` hiện tại | 9869 | 98,69% |
| Golden C đúng thứ tự ReLU → BN, chạy bằng harness kiểm toán | 9869 | 98,69% |
| Golden C hiện tại, BN → ReLU | 9037 | 90,37% |
| PTQ C hiện tại trong `w_qandcodeC` | 9051 | 90,51% |
| QAT C hiện tại trong `w_qandcodeC` | 9879 | 98,79% |

Thay thứ tự làm đổi **944 nhãn dự đoán**, giảm **832 ảnh đúng**, tương đương **8,32 điểm phần trăm**. Mức giảm chính đã có trong mô hình float, trước PTQ. PTQ hiện tại tăng 0,14 điểm so với Golden sai thứ tự, nên không thể quy mức giảm 98 → 90 cho sai số INT8.

Kết quả C sửa thứ tự được tạo bằng chính các operator của Golden C trong một harness tạm, chạy đủ 10000 ảnh rồi đối chiếu. Không chỉnh hoặc huấn luyện lại trọng số để thực hiện phép thử này. Cả bản C đúng và sai thứ tự đều khớp 10000/10000 nhãn với tham chiếu tương ứng.

## 2. Đối chiếu trọng số và dữ liệu

- Tất cả **14 tensor** TXT hiện tại khớp trọng số `.keras` sau khi xuất 6 chữ số thập phân, đúng với `fmt='%f'` trong notebook.
- Sai số phần tử lớn nhất khoảng **5,07×10^-7**. Tham chiếu float dùng TXT đúng thứ tự và Keras khớp toàn bộ 10000 nhãn. Làm tròn TXT hiện tại không gây mức giảm 8 điểm accuracy.
- Shape Conv1 `(3,3,1,8)`, Conv2 `(3,3,8,16)`, Dense `(400,10)` đúng. Flatten theo H,W,C; Conv HWIO; Dense IO được kiểm tra qua kết quả tương đương NumPy, TensorFlow và C.
- Epsilon BatchNorm `0.001` đúng với Keras. Pool VALID bỏ hàng/cột lẻ cuối đúng. Dropout phải tắt khi suy luận; bỏ Softmax khi chỉ lấy argmax không gây sai khác nhãn.
- `data/test_images.bin` giống chính xác MNIST test đã chia 255 và `test_labels.bin` giống chính xác nhãn MNIST test; đủ 10000 ảnh, không lệch ghép ảnh/nhãn.
- Các bản TXT có dấu cách đầu tên hiện có nội dung bằng bản không dấu cách. Chưa gây sai số trong lần kiểm tra này, nhưng cách đặt tên và chọn file theo thời gian sửa có thể dẫn tới trộn checkpoint ở lần xuất sau.

## 3. Các điểm cần sửa trong luồng dự án

### A. Golden C và notebook Inference sai thứ tự lớp — nguyên nhân trực tiếp

Trong `Source C/Golden Model AI.c`, vị trí dòng 177–179 và 183–185, cần đặt ReLU ngay sau Conv và trước BatchNorm:

```c
conv2d(img, conv1_out, w1, b1, 28, 28, 1, 8, 3, 3);
relu_inplace(conv1_out, 5408);
batch_norm_inplace(conv1_out, gamma1, beta1, mean1, var1, 5408, 8);
max_pool2d(conv1_out, pool1_out, 26, 26, 8);

conv2d(pool1_out, conv2_out, w2, b2, 13, 13, 8, 16, 3, 3);
relu_inplace(conv2_out, 1936);
batch_norm_inplace(conv2_out, gamma2, beta2, mean2, var2, 1936, 16);
max_pool2d(conv2_out, pool2_out, 11, 11, 16);
```

`notebooks/Inference_step.ipynb`, cell 0, cũng phải đổi `batch_norm`/`relu` ở cả hai khối. Kiểm tra một ảnh số 7 đúng không đủ chứng minh toàn bộ pipeline tương đương mô hình Keras.

### B. PTQ/QAT mới đang kế thừa Golden sai thứ tự

`w_qandcodeC/export_models.py`, hàm `golden_parameters()`, gộp BN vào Conv bằng `W*factor` và `(b-mean)*factor+beta`. Phép biến đổi này đúng với **Conv → BN → ReLU** của Golden đang có, nhưng không giữ được hàm số của **Conv → ReLU → BN** đã huấn luyện.

Do đó, PTQ hiện tại là lượng tử hóa chính xác theo Golden sai thứ tự. QAT đã tinh chỉnh trọng số trên chính kiến trúc sai thứ tự trong 3 epoch và đạt lại 98,79%. Kết quả QAT này không chứng minh phép gộp BN là tương đương mô hình huấn luyện gốc; mức tăng bao gồm tác dụng huấn luyện lại trên kiến trúc đã đổi.

Không thể chỉ đổi vài câu gọi trong C lượng tử hóa rồi giữ nguyên bộ trọng số đã gộp. Cần tái xuất PTQ/QAT theo đúng **Conv → ReLU → BN**, có scale cho dữ liệu âm sau BN. Có thể triển khai BN bằng affine hoặc threshold tương đương đã kiểm chứng; không gộp BN tuyến tính xuyên qua ReLU bằng công thức hiện tại. Để dùng cách gộp Conv→BN đơn giản, phải thiết kế/huấn luyện một mô hình có thứ tự đó từ đầu và đánh giá riêng.

### C. Notebook training hiện tại và checkpoint không cùng cấu hình

Mã notebook hiện tại pad ảnh lên **30×30** và `Input` đặt **30×30**. Mô hình `.keras` thực tế dùng **28×28**, Dense **400×10**. Kiến trúc notebook 30×30 sẽ cho 6×6×16=576 đặc trưng, Dense phải là **576×10**.

Thứ tự `execution_count` cũng không liên tục theo cell: cell dựng model là 29, cell fit là 15, cell save là 16, cell xuất trọng số là 10. Đây là bằng chứng notebook chứa trạng thái và output từ các lượt chạy khác nhau; không thể xem toàn bộ output lưu sẵn như một lần chạy đầu-cuối thống nhất.

Notebook hiển thị **98,88%**, nhưng checkpoint hiện có chạy lại đạt **98,69%**. Chưa có bằng chứng để gán chính xác chênh lệch 0,19 điểm này cho một lần huấn luyện cụ thể. Có thể khẳng định TXT hiện tại khớp checkpoint 98,69%; không thể khẳng định checkpoint đó chính là model của output 98,88% đã lưu trong notebook.

Cell save dùng `mnist_cnn_model.keras`, còn cell xuất dùng `../mnist_cnn_model.keras`; hai đường dẫn phụ thuộc thư mục làm việc của kernel. Cần thống nhất đường dẫn tuyệt đối từ gốc dự án, cấu hình 28 hoặc 30, rồi restart kernel và chạy toàn bộ notebook theo thứ tự nếu muốn tái huấn luyện.

### D. Cell xuất TXT có lỗi đặt tên file

Cell 20 xuất `" bn2_gamma.txt"`, `" bn2_beta.txt"`, `" bn2_mean.txt"`, `" bn2_var.txt"`, `" dense_weights.txt"`, `" dense_bias.txt"` với dấu cách đầu tên. Golden C mở tên không có dấu cách. Cần thống nhất tên, kiểm tra shape/số lượng và xuất cả manifest xác định model/đường dẫn; không chọn tensor tùy theo mtime của file.

### E. Script `notebooks/export_ptq.py` cũ không phải bộ xuất lượng tử hóa đầy đủ

- BN multipliers/shifts được đặt cố định `128`/`7`, không tính từ gamma/beta/mean/variance. Với cách dịch tương ứng, đây gần như là hệ số identity, không mô tả BN thực tế.
- Bias của mọi lớp mặc định dùng `scale_x=127`, dù activation của từng lớp cần scale riêng được xác định/calibrate.
- Ảnh INT8 được mã hóa `round(x*255-128)`, cần zero-point **-128**; không tương thích kernel đầu vào đối xứng zero-point 0 nếu không bù lại offset.
- Script không xuất đầy đủ scale/zero-point/requantization cho mỗi lớp.
- `data/test_images_int8.bin` hiện có **9000000 byte**, trong khi 10000 ảnh 28×28 INT8 phải có **7840000 byte**. File này không phải bản raw 28×28 tương ứng với float test hiện tại theo giao diện được mô tả. Kích thước 9000000 tương ứng 900 giá trị/ảnh nếu thật sự chứa 10000 ảnh, nhưng kiểm toán chưa kết luận nội dung đó chắc chắn là ảnh padding.

Hai bản C trong `w_qandcodeC` đọc file ảnh **float32** và tự lượng tử hóa; không sử dụng file INT8 sai kích thước này. Vì thế lỗi INT8 cũ là vấn đề riêng, không phải nguyên nhân trực tiếp của accuracy 90,51% đã đo.

### F. Các file kiểm tra khác

- `notebooks/infer_python.py` và `Source C/Time.c` có đúng thứ tự ReLU→BN; cả hai chạy lại đạt 98,69%. Chúng có thể dùng để đối chiếu Golden float.
- `notebooks/import time.py` dùng ảnh, nhãn và trọng số ngẫu nhiên, đồng thời chỉ có một khối Conv. Accuracy của file này không phải accuracy của mô hình MNIST đã huấn luyện.
- `Source C/mnist_PTQ.c` và `Source C/mnist_QAT.c` là các bản khác, đọc `artifacts/PTQ_fpga.bin`/`QAT_fpga.bin`; các model này hiện không có trên đĩa. Không nhầm chúng với hai bản trong `w_qandcodeC` khi biên dịch/chạy hoặc báo cáo accuracy.
- Golden C không kiểm tra giá trị trả về của `fscanf`, nên khi file sai/thiếu số có thể dùng dữ liệu chưa khởi tạo. Điều này không gây lần giảm accuracy đang khảo sát vì các TXT hiện tại đủ và khớp checkpoint, nhưng cần kiểm tra số lượng/giá trị khi nạp.

## 4. Trình tự khắc phục

1. Chọn checkpoint 28×28 đang có làm chuẩn hiện tại; kiểm chứng baseline Keras 98,69%.
2. Sửa thứ tự Golden C và notebook Inference để khớp ReLU→BN. Không cần huấn luyện lại để phục hồi baseline float: phép thử C đã chứng minh 98,69%.
3. Chuẩn hóa đường dẫn, tên file xuất TXT, shape và cấu hình notebook. Giữ được liên kết giữa từng bộ TXT và checkpoint gốc.
4. Sửa bộ xuất/kiến trúc lượng tử hóa theo đúng thứ tự, tái calibration bằng train, tái xuất PTQ và tinh chỉnh QAT. Các bộ lượng tử hiện tại không đại diện cho bản float đã sửa.
5. Dùng cùng 10000 ảnh/nhãn để so Keras → TXT float → C float → PTQ → QAT; không dùng test để chọn scale hoặc epoch. Báo cáo riêng từng bước.

Lần kiểm toán này ghi script, dự đoán đối chiếu và báo cáo vào `audit/`; các số đo trước đây được giữ để phân biệt nguyên nhân với kết quả sau khi sửa. Các phép chạy C hiện tại được kiểm tra lại bằng `verify_models.py`.
