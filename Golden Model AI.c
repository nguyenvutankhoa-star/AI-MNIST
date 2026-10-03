#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ==========================================
// 1. HÀM ĐỌC DỮ LIỆU TỪ FILE TXT
// ==========================================
void load_data(const char* filename, float* array, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Lỗi: Không thể mở file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        fscanf(file, "%f,", &array[i]);
    }
    fclose(file);
}

// ==========================================
// 2. CÁC TOÁN TỬ (OPERATOR) PHIÊN BẢN C THUẦN
// ==========================================

// 2.1. Phép tích chập Conv2D
void conv2d(float* input, float* output, float* weights, float* bias, 
            int h_in, int w_in, int c_in, int c_out, int k_h, int k_w) {
    int h_out = h_in - k_h + 1;
    int w_out = w_in - k_w + 1;
    
    for (int h = 0; h < h_out; h++) {
        for (int w = 0; w < w_out; w++) {
            for (int co = 0; co < c_out; co++) {
                float sum = bias[co];
                for (int kh = 0; kh < k_h; kh++) {
                    for (int kw = 0; kw < k_w; kw++) {
                        for (int ci = 0; ci < c_in; ci++) {
                            // Quy đổi mảng 3D/4D thành 1D
                            int in_idx = ((h + kh) * w_in + (w + kw)) * c_in + ci;
                            int w_idx = ((kh * k_w + kw) * c_in + ci) * c_out + co;
                            sum += input[in_idx] * weights[w_idx];
                        }
                    }
                }
                output[(h * w_out + w) * c_out + co] = sum;
            }
        }
    }
}

// 2.2. Chuẩn hóa theo lô (Chạy in-place để tiết kiệm RAM)
void batch_norm_inplace(float* data, float* gamma, float* beta, float* mean, float* var, int total_size, int channels) {
    float eps = 1e-3;
    int pixels = total_size / channels;
    for (int p = 0; p < pixels; p++) {
        for (int c = 0; c < channels; c++) {
            int idx = p * channels + c;
            data[idx] = gamma[c] * (data[idx] - mean[c]) / sqrt(var[c] + eps) + beta[c];
        }
    }
}

// 2.3. Hàm kích hoạt ReLU (Chạy in-place)
void relu_inplace(float* data, int total_size) {
    for (int i = 0; i < total_size; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

// 2.4. Lấy mẫu cực đại 2x2 (MaxPooling2D)
void max_pool2d(float* input, float* output, int h_in, int w_in, int channels) {
    int h_out = h_in / 2;
    int w_out = w_in / 2;
    
    for (int h = 0; h < h_out; h++) {
        for (int w = 0; w < w_out; w++) {
            for (int c = 0; c < channels; c++) {
                float max_val = -999999.0f; // Khởi tạo giá trị cực nhỏ
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int in_idx = ((h * 2 + kh) * w_in + (w * 2 + kw)) * channels + c;
                        if (input[in_idx] > max_val) {
                            max_val = input[in_idx];
                        }
                    }
                }
                output[(h * w_out + w) * channels + c] = max_val;
            }
        }
    }
}

// 2.5. Tầng kết nối đầy đủ (Dense)
void dense_layer(float* input, float* weights, float* bias, float* output, int in_size, int out_size) {
    for (int i = 0; i < out_size; i++) {
        output[i] = bias[i];
        for (int j = 0; j < in_size; j++) {
            output[i] += input[j] * weights[j * out_size + i];
        }
    }
}

// 2.6. Tìm vị trí Logit lớn nhất
int argmax(float* logits, int size) {
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ==========================================
// 3. HÀM MAIN - ĐIỀU PHỐI LUỒNG SUY LUẬN
// ==========================================
int main() {
    printf("--- KHOI DONG C/C++ GOLDEN MODEL ---\n");

    // ---------------------------------------------------------
    // BƯỚC 1: CẤP PHÁT BỘ NHỚ CHO TRỌNG SỐ VÀ DỮ LIỆU TRUNG GIAN
    // ---------------------------------------------------------
    float *img = (float*)malloc(28 * 28 * 1 * sizeof(float));
    
    // Lớp 1 (Conv -> BN -> ReLU -> Pool)
    float *w1 = (float*)malloc(3 * 3 * 1 * 8 * sizeof(float));
    float *b1 = (float*)malloc(8 * sizeof(float));
    float *gamma1 = (float*)malloc(8 * sizeof(float));
    float *beta1 = (float*)malloc(8 * sizeof(float));
    float *mean1 = (float*)malloc(8 * sizeof(float));
    float *var1 = (float*)malloc(8 * sizeof(float));
    
    float *conv1_out = (float*)malloc(26 * 26 * 8 * sizeof(float));
    float *pool1_out = (float*)malloc(13 * 13 * 8 * sizeof(float));

    // Lớp 2 (Conv -> BN -> ReLU -> Pool)
    float *w2 = (float*)malloc(3 * 3 * 8 * 16 * sizeof(float));
    float *b2 = (float*)malloc(16 * sizeof(float));
    float *gamma2 = (float*)malloc(16 * sizeof(float));
    float *beta2 = (float*)malloc(16 * sizeof(float));
    float *mean2 = (float*)malloc(16 * sizeof(float));
    float *var2 = (float*)malloc(16 * sizeof(float));
    
    float *conv2_out = (float*)malloc(11 * 11 * 16 * sizeof(float));
    float *pool2_out = (float*)malloc(5 * 5 * 16 * sizeof(float)); // Chứa 400 phần tử (Flatten)

    // Lớp Dense
    float *w_dense = (float*)malloc(400 * 10 * sizeof(float));
    float *b_dense = (float*)malloc(10 * sizeof(float));
    float logits[10];

    // ---------------------------------------------------------
    // BƯỚC 2: NẠP TRỌNG SỐ TỪ FILE
    // ---------------------------------------------------------
    printf("Dang nap trong so...\n");
    load_data("input_image.txt", img, 784); // Ảnh đầu vào
    
    load_data("conv1_weights.txt", w1, 72); load_data("conv1_bias.txt", b1, 8);
    load_data("bn1_gamma.txt", gamma1, 8);  load_data("bn1_beta.txt", beta1, 8);
    load_data("bn1_mean.txt", mean1, 8);    load_data("bn1_var.txt", var1, 8);

    load_data("conv2_weights.txt", w2, 1152); load_data("conv2_bias.txt", b2, 16);
    load_data("bn2_gamma.txt", gamma2, 16);   load_data("bn2_beta.txt", beta2, 16);
    load_data("bn2_mean.txt", mean2, 16);     load_data("bn2_var.txt", var2, 16);

    load_data("dense_weights.txt", w_dense, 4000); load_data("dense_bias.txt", b_dense, 10);

    // ---------------------------------------------------------
    // BƯỚC 3: THỰC THI LAN TRUYỀN THUẬN (INFERENCE)
    // ---------------------------------------------------------
    printf("Dang chay suy luan...\n");
    
    // Khối 1
    conv2d(img, conv1_out, w1, b1, 28, 28, 1, 8, 3, 3);
    batch_norm_inplace(conv1_out, gamma1, beta1, mean1, var1, 5408, 8);
    relu_inplace(conv1_out, 5408);
    max_pool2d(conv1_out, pool1_out, 26, 26, 8);

    // Khối 2
    conv2d(pool1_out, conv2_out, w2, b2, 13, 13, 8, 16, 3, 3);
    batch_norm_inplace(conv2_out, gamma2, beta2, mean2, var2, 1936, 16);
    relu_inplace(conv2_out, 1936);
    max_pool2d(conv2_out, pool2_out, 11, 11, 16);

    // Khối Dense (400 đặc trưng -> 10 nhãn)
    dense_layer(pool2_out, w_dense, b_dense, logits, 400, 10);

    // ---------------------------------------------------------
    // BƯỚC 4: IN KẾT QUẢ
    // ---------------------------------------------------------
    printf("\nGia tri Logits:\n");
    for(int i = 0; i < 10; i++) {
        printf("Logit[%d] = %f\n", i, logits[i]);
    }

    int predicted_label = argmax(logits, 10);
    printf("\n=> KET QUA DU DOAN CUA GOLDEN MODEL: %d\n", predicted_label);

    // Dọn dẹp RAM
    free(img); free(w1); free(b1); free(gamma1); free(beta1); free(mean1); free(var1);
    free(conv1_out); free(pool1_out);
    free(w2); free(b2); free(gamma2); free(beta2); free(mean2); free(var2);
    free(conv2_out); free(pool2_out);
    free(w_dense); free(b_dense);

    return 0;
}