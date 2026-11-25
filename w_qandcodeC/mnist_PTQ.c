/* PTQ: lượng tử hóa sau huấn luyện, không cập nhật trọng số bằng gradient.
 * V2: đúng checkpoint Keras: Conv -> ReLU -> BN -> Pool (hai khối) -> Dense.
 * BN được biến đổi thành affine kèm ngưỡng BN(0), KHÔNG chặn tại 0 sau BN.
 * Với alpha=gamma/sqrt(var+0.001), t=beta-alpha*mean, z=alpha*Conv+t:
 * alpha*ReLU(Conv)+t = max(z,t) nếu alpha>=0; = min(z,t) nếu alpha<0.
 * Layout ảnh NHWC; kernel HWIO; Dense IO; flatten giữ thứ tự H,W,C.
 * Quy ước x_thực = q*2^exponent, zero-point=0, miền INT8 [-127,127].
 * Làm tròn gần nhất, hòa thì ra xa 0; weights INT8, bias/accumulator INT32.
 * QAT dùng cùng kernel suy luận; sự khác biệt nằm ở quá trình tạo trọng số.
 * Biên dịch: gcc -O3 -std=c11 "w_qandcodeC/mnist_PTQ.c" -lm -o "w_qandcodeC/mnist_PTQ.exe"
 * Không tham số: đánh giá TOÀN BỘ 10000 ảnh test, giống --test.
 * Một tham số: đường dẫn ảnh TXT chứa đúng 784 giá trị float trong [0,1].
 * --test [data_dir] [predictions.bin]: đánh giá đúng 10000 ảnh test float32/nhãn int32.
 * Trọng số được nhúng từ PTQ/weights.h; không cần mở TXT/BIN khi chạy.
 */
#include <stdio.h>     /* FILE, fopen, fread, printf và thông báo lỗi. */
#include <stdlib.h>    /* exit, EXIT_FAILURE và EXIT_SUCCESS. */
#include <stdint.h>    /* int8_t, int32_t, int64_t có số bit xác định. */
#include <string.h>    /* strcmp để nhận diện lựa chọn --test. */
#include <math.h>      /* isfinite, round, ldexp cho chuyển đổi ở biên vào/ra. */
#include <time.h>      /* clock đo thời gian CPU của quá trình đánh giá. */
#include <limits.h>    /* INT32_MIN để khởi tạo tìm logit lớn nhất. */
#include "PTQ/weights.h" /* Mảng trọng số INT8, bias INT32 và số mũ scale. */
#define MODEL_NAME "PTQ" /* Nhãn mô hình được in trong kết quả. */

static int8_t image_q[784]; /* Ảnh đầu vào lượng tử hóa, 28*28*1 phần tử. */
static int8_t pool1_q[1352]; /* Đầu ra pool1: 13*13*8 phần tử INT8. */
static int8_t pool2_q[400]; /* Đầu ra pool2: 5*5*16, cũng là đầu vào Dense. */
static int32_t logits_q[10]; /* Mười accumulator Dense, cùng một scale nên argmax trực tiếp. */

static void fail(const char *message) { /* Dừng ngay khi dữ liệu hoặc đường dẫn sai. */
    fprintf(stderr, "ERROR: %s\n", message); /* Gửi nguyên nhân lỗi sang stderr. */
    exit(EXIT_FAILURE); /* Trả mã lỗi để script gọi phát hiện thất bại. */
} /* Kết thúc hàm báo lỗi. */

static int8_t quantize_input(float x) { /* Chuyển pixel float [0,1] thành INT8. */
    if (!isfinite(x) || x < 0.0f || x > 1.0f) fail("pixel must be finite and in [0,1]"); /* Kiểm tra miền dữ liệu. */
    double q = round(ldexp((double)x, -INPUT_EXP)); /* Chia scale 2^INPUT_EXP rồi làm tròn hòa ra xa 0. */
    if (q > 127.0) q = 127.0; /* Bão hòa phía trên để không vượt INT8. */
    if (q < -127.0) q = -127.0; /* Bão hòa phía dưới theo quy ước đối xứng. */
    return (int8_t)q; /* Ép kiểu sau khi bảo đảm giá trị nằm trong miền hợp lệ. */
} /* Kết thúc lượng tử hóa đầu vào. */

static int8_t requantize_signed(int32_t accumulator, int shift) { /* Đổi scale, giữ giá trị âm hợp lệ sau BN. */
    int64_t value = accumulator; /* Dùng INT64 cho bước đổi scale để tránh tràn trung gian. */
    if (shift >= 0) { /* Trường hợp scale accumulator lớn hơn hoặc bằng scale output. */
        if (shift > 30) fail("invalid positive shift"); /* Chặn số mũ ngoài giới hạn triển khai. */
        value *= INT64_C(1) << shift; /* Nhân với 2^shift bằng số nguyên không mất phần lẻ. */
    } else { /* Trường hợp cần chia để đưa accumulator về scale output. */
        int bits = -shift; /* Đổi số mũ âm thành số bit cần chia. */
        if (bits > 60) fail("invalid negative shift"); /* Giữ phép dịch trong miền INT64 hợp lệ. */
        int64_t divisor = INT64_C(1) << bits; /* Tạo mẫu số 2^bits bằng kiểu 64 bit. */
        value = value >= 0 ? (value + divisor/2)/divisor : -((-value + divisor/2)/divisor); /* Làm tròn gần nhất, hòa ra xa 0 cho cả hai dấu. */
    } /* Hoàn tất đổi scale bằng số nguyên. */
    return (int8_t)(value > 127 ? 127 : value < -127 ? -127 : value); /* Bão hòa đối xứng [-127,127], không loại giá trị âm sau BN. */
} /* Kết thúc requantization có dấu. */

static void conv_relu_pool(const int8_t *input, int8_t *output, const int8_t *weights,
                           const int32_t *bias, const int32_t *threshold, const int32_t *direction,
                           int width, int ci, int co, int shift) { /* Conv3x3 -> ReLU -> BN -> Pool bằng biến đổi ngưỡng tương đương. */
    int pooled_width = (width - 2) / 2; /* 28->26->13 ở khối 1; 13->11->5 ở khối 2. */
    for (int ph = 0; ph < pooled_width; ++ph) { /* Duyệt hàng của feature map sau pooling. */
        for (int pw = 0; pw < pooled_width; ++pw) { /* Duyệt cột của feature map sau pooling. */
            for (int oc = 0; oc < co; ++oc) { /* Duyệt từng kênh đầu ra. */
                int8_t maximum = -127; /* BN nằm sau ReLU nên cửa sổ pooling có thể chứa toàn số âm. */
                for (int dh = 0; dh < 2; ++dh) { /* Duyệt hai hàng trong cửa sổ pooling. */
                    for (int dw = 0; dw < 2; ++dw) { /* Duyệt hai cột trong cửa sổ pooling. */
                        int32_t sum = bias[oc]; /* Bias có scale input_scale*weight_scale. */
                        for (int kh = 0; kh < 3; ++kh) { /* Duyệt hàng kernel 3x3. */
                            for (int kw = 0; kw < 3; ++kw) { /* Duyệt cột kernel 3x3. */
                                for (int ic = 0; ic < ci; ++ic) { /* Tích lũy qua tất cả kênh đầu vào. */
                                    int ix = ((2*ph + dh + kh)*width + 2*pw + dw + kw)*ci + ic; /* Chỉ số ảnh NHWC phẳng. */
                                    int iw = ((kh*3 + kw)*ci + ic)*co + oc; /* Chỉ số kernel HWIO giống Golden Model. */
                                    sum += (int32_t)input[ix] * (int32_t)weights[iw]; /* MAC INT8*INT8 vào INT32; bộ xuất đã kiểm tra cận tràn. */
                                } /* Kết thúc tổng theo kênh vào. */
                            } /* Kết thúc cột kernel. */
                        } /* Kết thúc hàng kernel. */
                        if (direction[oc] > 0 && sum < threshold[oc]) sum = threshold[oc]; /* alpha>=0: max(z,BN(0)) biểu diễn đúng BN(ReLU(Conv)). */
                        if (direction[oc] < 0 && sum > threshold[oc]) sum = threshold[oc]; /* alpha<0: dùng min(z,BN(0)), không được dùng max. */
                        int8_t candidate = requantize_signed(sum, shift); /* Đưa activation BN có dấu về scale đầu vào lớp kế tiếp. */
                        if (candidate > maximum) maximum = candidate; /* Giữ giá trị lớn nhất của cửa sổ 2x2. */
                    } /* Kết thúc cột pooling. */
                } /* Kết thúc hàng pooling. */
                output[(ph*pooled_width + pw)*co + oc] = maximum; /* Ghi trực tiếp pool output, tiết kiệm feature map Conv. */
            } /* Kết thúc kênh đầu ra. */
        } /* Kết thúc cột đầu ra. */
    } /* Kết thúc hàng đầu ra; hàng/cột Conv thứ 11 bị bỏ đúng như VALID Pool của Golden. */
} /* Kết thúc khối Conv, ReLU, BN và Pool hợp nhất đúng thứ tự checkpoint. */

static int infer(void) { /* Chạy mạng từ image_q và trả nhãn 0..9. */
    conv_relu_pool(image_q, pool1_q, w1, b1, t1, d1, 28, 1, 8, SHIFT1); /* Khối 1 dùng ngưỡng t1=BN1(0), chiều chặn d1. */
    conv_relu_pool(pool1_q, pool2_q, w2, b2, t2, d2, 13, 8, 16, SHIFT2); /* Khối 2 dùng ngưỡng t2=BN2(0), giữ cả activation âm. */
    int best = 0; /* Chọn nhãn nhỏ nhất khi có các logit bằng nhau, giống Golden argmax. */
    int32_t maximum = INT32_MIN; /* Khởi tạo nhỏ hơn hoặc bằng mọi giá trị INT32 hợp lệ. */
    for (int oc = 0; oc < 10; ++oc) { /* Tính logit của từng chữ số. */
        int32_t sum = b3[oc]; /* Bias Dense đã lượng tử hóa theo scale đầu vào Dense nhân scale weight. */
        for (int ic = 0; ic < 400; ++ic) { /* Duyệt 400 đặc trưng flatten theo thứ tự H,W,C. */
            sum += (int32_t)pool2_q[ic] * (int32_t)w3[ic*10 + oc]; /* Tích vô hướng theo layout IO của Golden. */
        } /* Kết thúc tổng Dense của một nhãn. */
        logits_q[oc] = sum; /* Lưu logit nguyên để có thể in hoặc giải lượng tử. */
        if (sum > maximum) { /* Chỉ cập nhật khi lớn hơn để bảo toàn quy tắc hòa. */
            maximum = sum; /* Ghi nhận logit lớn nhất hiện tại. */
            best = oc; /* Ghi nhận nhãn tương ứng với logit lớn nhất. */
        } /* Kết thúc cập nhật argmax. */
    } /* Kết thúc mười đầu ra Dense. */
    return best; /* Không cần Softmax vì không làm thay đổi thứ tự logits. */
} /* Kết thúc suy luận hoàn toàn số nguyên sau bước lượng tử hóa ảnh. */

static FILE *open_data(const char *name, const char *directory) { /* Mở dữ liệu ở thư mục người dùng chỉ định hoặc vị trí mặc định. */
    const char *candidates[] = {"data", "../data"}; /* Hỗ trợ chạy từ gốc dự án hoặc từ w_qandcodeC/Source C. */
    char path[2048]; /* Buffer chứa đường dẫn đầy đủ, có kiểm tra độ dài. */
    int tries = directory ? 1 : 2; /* Có đường dẫn rõ ràng thì không tự chuyển sang thư mục khác. */
    for (int i = 0; i < tries; ++i) { /* Thử từng thư mục theo thứ tự. */
        const char *base = directory ? directory : candidates[i]; /* Chọn thư mục cho lần thử hiện tại. */
        int length = snprintf(path, sizeof path, "%s/%s", base, name); /* Ghép đường dẫn an toàn có giới hạn buffer. */
        if (length < 0 || length >= (int)sizeof path) fail("data path too long"); /* Chặn đường dẫn bị cắt ngắn. */
        FILE *file = fopen(path, "rb"); /* Mở nhị phân để tránh Windows dịch byte newline. */
        if (file) return file; /* Trả file ngay khi tìm thấy dữ liệu. */
    } /* Kết thúc các vị trí tìm kiếm. */
    fail(name); /* In tên dữ liệu thiếu và dừng chương trình. */
    return NULL; /* Nhánh hình thức để thỏa kiểu trả về; fail luôn kết thúc tiến trình. */
} /* Kết thúc mở dữ liệu. */

static void read_exact(FILE *file, void *buffer, size_t bytes) { /* Đọc đúng số byte cho một bản ghi. */
    if (fread(buffer, 1, bytes, file) != bytes) fail("input truncated or read failed"); /* Không dùng dữ liệu chưa được nạp đủ. */
} /* Kết thúc hàm đọc có kiểm tra. */

static void check_end(FILE *file) { /* Xác nhận không còn dữ liệu thừa hoặc lỗi I/O. */
    if (fgetc(file) != EOF || ferror(file)) fail("unexpected trailing data or I/O error"); /* Chặn tệp sai số lượng bản ghi. */
} /* Kết thúc kiểm tra cuối file. */

static void load_text_image(FILE *file) { /* Nạp ảnh TXT phân tách bằng khoảng trắng hoặc dấu phẩy. */
    for (int i = 0; i < 784; ++i) { /* Cần đúng 28*28 giá trị pixel. */
        float pixel; /* Lưu pixel float trước lượng tử hóa. */
        if (fscanf(file, " %f", &pixel) != 1) fail("image must contain 784 float pixels"); /* Phát hiện thiếu hoặc sai định dạng. */
        image_q[i] = quantize_input(pixel); /* Kiểm tra miền và chuyển pixel sang INT8. */
        int next = fgetc(file); /* Đọc ký tự sau số để chấp nhận dấu phẩy tùy chọn. */
        if (next != ',' && next != EOF) ungetc(next, file); /* Trả lại khoảng trắng/ký tự để lượt đọc sau xử lý. */
    } /* Hoàn tất 784 pixel. */
    int tail; /* Biến lưu ký tự kiểm tra phần còn lại của file. */
    do { tail = fgetc(file); } while (tail == ' ' || tail == '\t' || tail == '\r' || tail == '\n'); /* Bỏ khoảng trắng cuối tệp. */
    if (tail != EOF || ferror(file)) fail("unexpected data after image"); /* Không chấp nhận ảnh có pixel thừa. */
} /* Kết thúc nạp ảnh TXT. */

int main(int argc, char **argv) { /* Điểm bắt đầu của chương trình C độc lập. */
    uint32_t endian = 1; /* Giá trị dùng kiểm tra thứ tự byte trên máy đang chạy. */
    if (sizeof(float) != 4 || *(unsigned char *)&endian != 1) fail("requires little-endian float32 host"); /* BIN hiện có dùng float32/int32 little-endian. */
    setvbuf(stdout, NULL, _IOLBF, 0); /* Đẩy thông báo ngay sau mỗi dòng để theo dõi tiến độ. */
    printf("%s V2: Conv-ReLU-BN, signed BN(0) thresholds; default=10000 test images\n", MODEL_NAME); /* Nhận diện bản đã sửa để tránh chạy nhầm EXE cũ. */
    if (argc == 1 || (argc >= 2 && strcmp(argv[1], "--test") == 0)) { /* Mặc định hoặc --test đều đánh giá toàn bộ 10000 ảnh. */
        if (argc > 4) fail("usage: program --test [data_dir] [predictions.bin]"); /* Từ chối tham số ngoài giao diện đã mô tả. */
        const char *directory = argc >= 3 ? argv[2] : NULL; /* Cho phép người dùng chỉ rõ thư mục dữ liệu. */
        FILE *images = open_data("test_images.bin", directory); /* Mở 10000 ảnh float32 [0,1]. */
        FILE *labels = open_data("test_labels.bin", directory); /* Mở 10000 nhãn int32 0..9. */
        FILE *predictions = argc == 4 ? fopen(argv[3], "wb") : NULL; /* Chỉ tạo file dự đoán khi được yêu cầu bằng tham số. */
        if (argc == 4 && !predictions) fail(argv[3]); /* Dừng nếu không thể ghi đường dẫn dự đoán. */
        int correct = 0; /* Bộ đếm số ảnh được phân loại đúng. */
        double start = (double)clock() / CLOCKS_PER_SEC; /* Ghi thời điểm CPU trước vòng lặp đánh giá. */
        for (int n = 0; n < 10000; ++n) { /* Đánh giá đúng 10000 ảnh, không huấn luyện hoặc calibration ở đây. */
            float pixels[784]; /* Buffer một ảnh float32 từ tập test. */
            int32_t label; /* Nhãn thật của ảnh đang xử lý. */
            read_exact(images, pixels, sizeof pixels); /* Đọc đủ 3136 byte pixel. */
            read_exact(labels, &label, sizeof label); /* Đọc đủ 4 byte nhãn. */
            if (label < 0 || label > 9) fail("label outside 0..9"); /* Phát hiện nhãn sai miền hoặc sai định dạng nhị phân. */
            for (int i = 0; i < 784; ++i) image_q[i] = quantize_input(pixels[i]); /* Đưa toàn ảnh về scale input của mô hình. */
            int32_t prediction = infer(); /* Chạy hai khối Conv và Dense bằng số nguyên. */
            correct += prediction == label; /* Cộng một nếu dự đoán trùng nhãn thật. */
            if (predictions && fwrite(&prediction, sizeof prediction, 1, predictions) != 1) fail("prediction write failed"); /* Lưu nhãn int32 để đối chiếu với Python. */
            if ((n + 1) % 1000 == 0) printf("%s: %d/10000, accuracy=%.4f%%\n", MODEL_NAME, n+1, 100.0*correct/(n+1)); /* Báo tiến độ mỗi 1000 ảnh. */
        } /* Hoàn tất toàn bộ tập test. */
        check_end(images); /* Bảo đảm không có ảnh thừa sau ảnh thứ 10000. */
        check_end(labels); /* Bảo đảm không có nhãn thừa sau nhãn thứ 10000. */
        fclose(images); /* Đóng file ảnh sau đánh giá. */
        fclose(labels); /* Đóng file nhãn sau đánh giá. */
        if (predictions && fclose(predictions) != 0) fail("prediction close failed"); /* Kiểm tra việc flush file dự đoán xuống đĩa. */
        printf("%s correct=%d/10000 accuracy=%.4f%% CPU_seconds=%.3f\n", MODEL_NAME, correct, 100.0*correct/10000, (double)clock()/CLOCKS_PER_SEC-start); /* In kết quả tổng hợp. */
    } else { /* Khi truyền đường dẫn ảnh TXT, chỉ suy luận ảnh đó. */
        if (argc > 2) fail("usage: program [image.txt] OR program --test [data_dir] [predictions.bin]"); /* Kiểm tra số đối số chế độ một ảnh. */
        FILE *image = argc == 2 ? fopen(argv[1], "r") : open_data("input_image.txt", NULL); /* Mở ảnh được chỉ định hoặc ảnh mặc định. */
        if (!image) fail(argv[1]); /* Báo lỗi nếu ảnh tùy chọn không mở được. */
        load_text_image(image); /* Nạp 784 pixel và lượng tử hóa đầu vào. */
        fclose(image); /* Đóng file ảnh vì toàn bộ dữ liệu đã nằm trong RAM. */
        int prediction = infer(); /* Chạy mạng và lấy nhãn có logit lớn nhất. */
        for (int c = 0; c < 10; ++c) printf("Logit[%d] INT32=%d real=%.6f\n", c, (int)logits_q[c], ldexp((double)logits_q[c], LOGIT_EXP)); /* Giải lượng tử chỉ để hiển thị logit. */
        printf("%s prediction: %d\n", MODEL_NAME, prediction); /* In chữ số được mô hình nhận dạng. */
    } /* Kết thúc lựa chọn chế độ chạy. */
    return EXIT_SUCCESS; /* Trả mã thành công sau khi hoàn thành suy luận/đánh giá. */
} /* Kết thúc main. */
