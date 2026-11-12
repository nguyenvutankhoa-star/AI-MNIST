import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def infer_batch_python(x_batch, w_conv, b_conv, w_dense, b_dense):
    # 1. Conv2D (N, 28, 28) -> (N, 26, 26, 8)
    windows = sliding_window_view(x_batch, (3, 3), axis=(1, 2)) 
    conv_out = np.tensordot(windows, w_conv, axes=((3, 4), (1, 2))) + b_conv
    
    # 2. ReLU
    np.maximum(conv_out, 0, out=conv_out)
    
    # 3. MaxPool 2x2 (N, 26, 26, 8) -> (N, 13, 13, 8)
    pool_out = conv_out.reshape(x_batch.shape[0], 13, 2, 13, 2, 8).max(axis=(2, 4))
    
    # 4. Flatten & Dense (N, 1352) -> (N, 10)
    flat = pool_out.reshape(x_batch.shape[0], -1)
    logits = flat @ w_dense + b_dense
    
    return logits

if __name__ == "__main__":
    BATCH_SIZE = 10000
    
    # Khởi tạo Dummy Data và Nhãn thực tế ngẫu nhiên (từ 0 đến 9)
    np.random.seed(42)
    x_batch = np.random.rand(BATCH_SIZE, 28, 28).astype(np.float32)
    y_true = np.random.randint(0, 10, size=BATCH_SIZE) # 10000 nhãn thực tế
    
    # Khởi tạo Trọng số ngẫu nhiên (Thay bằng load_model trong thực tế)
    w_conv = np.random.rand(8, 3, 3).astype(np.float32)
    b_conv = np.random.rand(8).astype(np.float32)
    w_dense = np.random.rand(1352, 10).astype(np.float32)
    b_dense = np.random.rand(10).astype(np.float32)

    # Chạy làm nóng hệ thống (Warm-up)
    _ = infer_batch_python(x_batch[:10], w_conv, b_conv, w_dense, b_dense)

    # --- BẮT ĐẦU ĐO THỜI GIAN SUY LUẬN ---
    start_time = time.perf_counter()
    
    logits_batch = infer_batch_python(x_batch, w_conv, b_conv, w_dense, b_dense)
        
    end_time = time.perf_counter()
    # --- KẾT THÚC ĐO THỜI GIAN ---
    
    # Đánh giá độ chính xác (Argmax)
    predictions = np.argmax(logits_batch, axis=1) # Tìm class có logit lớn nhất
    correct = np.sum(predictions == y_true)
    incorrect = BATCH_SIZE - correct
    accuracy = (correct / BATCH_SIZE) * 100
    
    total_time_ms = (end_time - start_time) * 1000
    
    print(f"========== BÁO CÁO PYTHON (NUMPY) ==========")
    print(f"Tổng số ảnh kiểm tra : {BATCH_SIZE} ảnh")
    print(f"Dự đoán ĐÚNG         : {correct} ảnh")
    print(f"Dự đoán SAI          : {incorrect} ảnh")
    print(f"Tỉ lệ chính xác      : {accuracy:.2f}%")
    print(f"--------------------------------------------")
    print(f"Tổng thời gian chạy  : {total_time_ms:.2f} ms")
    print(f"Thời gian/1 ảnh      : {total_time_ms / BATCH_SIZE:.5f} ms")
    print(f"============================================")