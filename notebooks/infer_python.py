import os
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ============================================================================
# KIẾN TRÚC (khớp 100% với Training_step.ipynb):
#   Input -> Conv2D(8,3x3,relu) -> BN -> MaxPool2 -> Conv2D(16,3x3,relu) -> BN
#         -> MaxPool2 -> Flatten -> (Dropout: tắt khi suy luận) -> Dense(10)
# LƯU Ý: BatchNorm nằm SAU ReLU nên KHÔNG gộp được vào Conv -> tính riêng.
# Keras BatchNormalization mặc định epsilon = 1e-3 (không phải 1e-5).
# ============================================================================
BN_EPS = 1e-3
NUM_CLASSES = 10
IMG = 28  # kích thước ảnh MNIST gốc

# --- Thư mục tìm file (thử lần lượt, trộn cả đường dẫn tuyệt đối và tương đối) ---
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # chạy trong notebook
    HERE = os.getcwd()

_dirs = [
    r"D:\reports\weights",
    r"D:\reports\data",
    os.path.join(HERE, "..", "weights"),
    os.path.join(HERE, "..", "data"),
    os.path.join(HERE, "weights"),
    os.path.join(HERE, "data"),
    HERE,
    os.path.join(os.getcwd(), "..", "weights"),
    os.path.join(os.getcwd(), "weights"),
    os.getcwd(),
]
SEARCH_DIRS = []
for _d in _dirs:
    _d = os.path.abspath(_d)
    if _d not in SEARCH_DIRS:
        SEARCH_DIRS.append(_d)
WEIGHTS_DIR = SEARCH_DIRS[0]


def find_file(filename):
    """Tìm file trong SEARCH_DIRS.

    Chấp nhận tên có dấu cách thừa ở đầu/cuối (vd ' bn2_gamma.txt', do code export
    trong notebook lưu nhầm tên có khoảng trắng). Nếu có nhiều bản trùng tên,
    chọn bản MỚI NHẤT (tránh đọc nhầm file cũ còn sót lại).
    """
    target = filename.strip().lower()
    best, best_mtime = None, -1.0
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.strip().lower() == target:
                p = os.path.join(d, f)
                if os.path.isfile(p) and os.path.getmtime(p) > best_mtime:
                    best, best_mtime = p, os.path.getmtime(p)
    return best


def load_flat(filename, expected=None):
    """Đọc file txt (mỗi dòng 1 số) thành mảng float32 1 chiều và kiểm tra số phần tử."""
    path = find_file(filename)
    if path is None:
        raise FileNotFoundError(
            f"Không tìm thấy '{filename}'. Đã tìm trong:\n  " + "\n  ".join(SEARCH_DIRS)
        )
    data = np.loadtxt(path, dtype=np.float32, delimiter=",").reshape(-1)
    if expected is not None and data.size != expected:
        raise ValueError(
            f"'{path}' có {data.size} giá trị, cần {expected}. "
            f"(File cũ/sai kiến trúc? Hãy chạy lại cell export trong Training_step.ipynb)"
        )
    print(f"   {os.path.basename(path).strip():<18s} <- {path}")
    return data


def load_model_params():
    print(f"Đang tìm các file trọng số (ưu tiên {WEIGHTS_DIR}) ...")
    p = {}

    # Keras lưu Conv2D kernel dạng (kh, kw, Cin, Cout) -> duỗi C-order
    # => reshape (kh*kw*Cin, Cout) là đúng thứ tự im2col (kh, kw, cin)
    p["w1"] = load_flat("conv1_weights.txt", 3 * 3 * 1 * 8).reshape(9, 8)
    p["b1"] = load_flat("conv1_bias.txt", 8)
    g, b, m, v = (load_flat(f"bn1_{n}.txt", 8) for n in ("gamma", "beta", "mean", "var"))
    p["sc1"] = (g / np.sqrt(v + BN_EPS)).astype(np.float32)
    p["sh1"] = (b - m * p["sc1"]).astype(np.float32)

    p["w2"] = load_flat("conv2_weights.txt", 3 * 3 * 8 * 16).reshape(72, 16)
    p["b2"] = load_flat("conv2_bias.txt", 16)
    g, b, m, v = (load_flat(f"bn2_{n}.txt", 16) for n in ("gamma", "beta", "mean", "var"))
    p["sc2"] = (g / np.sqrt(v + BN_EPS)).astype(np.float32)
    p["sh2"] = (b - m * p["sc2"]).astype(np.float32)

    wd = load_flat("dense_weights.txt")
    p["bd"] = load_flat("dense_bias.txt", NUM_CLASSES)
    if wd.size % NUM_CLASSES:
        raise ValueError(f"dense_weights.txt có {wd.size} giá trị, không chia hết cho 10")
    n_in = wd.size // NUM_CLASSES
    p["wd"] = wd.reshape(n_in, NUM_CLASSES)

    # Suy ra kích thước ảnh đầu vào (có/không padding) từ số đầu vào của Dense
    p["pad"] = None
    for pad in range(0, 4):
        s = IMG + 2 * pad
        s = ((s - 2) // 2 - 2) // 2
        if s * s * 16 == n_in:
            p["pad"] = pad
            break
    if p["pad"] is None:
        raise ValueError(f"Dense có {n_in} đầu vào, không khớp kiến trúc (400 hoặc 576).")
    print(f"-> Dense: {n_in} đầu vào => ảnh vào mạng {IMG + 2 * p['pad']}x{IMG + 2 * p['pad']} "
          f"(padding={p['pad']})")
    return p


def maxpool2(a):
    n, h, w, c = a.shape
    h2, w2 = h // 2, w // 2  # giống Keras 'valid': bỏ hàng/cột lẻ cuối
    return a[:, : 2 * h2, : 2 * w2, :].reshape(n, h2, 2, w2, 2, c).max(axis=(2, 4))


def infer_batch_python(x, p):
    """x: (N, H, W) float32 -> logits (N, 10)"""
    n = x.shape[0]

    # 1. Conv1 + ReLU + BN
    win = sliding_window_view(x, (3, 3), axis=(1, 2))               # (N,h,w,3,3)
    h1, w1 = win.shape[1], win.shape[2]
    a = win.reshape(n, h1, w1, 9) @ p["w1"] + p["b1"]               # (N,h,w,8)
    np.maximum(a, 0, out=a)
    a *= p["sc1"]
    a += p["sh1"]
    a = maxpool2(a)

    # 2. Conv2 + ReLU + BN
    win = sliding_window_view(a, (3, 3), axis=(1, 2))               # (N,h,w,C,3,3)
    h2, w2 = win.shape[1], win.shape[2]
    win = win.transpose(0, 1, 2, 4, 5, 3).reshape(n, h2, w2, 72)    # thứ tự (kh,kw,cin)
    a = win @ p["w2"] + p["b2"]
    np.maximum(a, 0, out=a)
    a *= p["sc2"]
    a += p["sh2"]
    a = maxpool2(a)

    # 3. Flatten (thứ tự H,W,C như Keras) + Dense (Dropout bỏ qua khi suy luận)
    return a.reshape(n, -1) @ p["wd"] + p["bd"]


def run_chunked(x, p, chunk=500):
    """Chạy theo từng cụm ảnh để không tốn quá nhiều RAM."""
    out = [infer_batch_python(x[i:i + chunk], p) for i in range(0, x.shape[0], chunk)]
    return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------------
# Dữ liệu test
# ----------------------------------------------------------------------------
def _to_images(flat, n):
    """flat -> (n, 28, 28); nếu là ảnh đã padding (vd 30x30) thì cắt viền."""
    side = int(round(np.sqrt(flat.size / n)))
    if side * side * n != flat.size or side < IMG:
        raise ValueError(f"Dữ liệu ảnh {flat.size} giá trị không khớp {n} ảnh.")
    x = flat.reshape(n, side, side)
    c = (side - IMG) // 2
    x = x[:, c:c + IMG, c:c + IMG]
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.max() > 1.5:  # đang ở thang 0..255
        x /= 255.0
    return x


def load_test_data():
    # a) file .bin (nhanh nhất): images float32 [0,1], labels int32
    pi, pl = find_file("test_images.bin"), find_file("test_labels.bin")
    if pi and pl:
        y = np.fromfile(pl, dtype=np.int32)
        x = _to_images(np.fromfile(pi, dtype=np.float32), y.size)
        print(f"-> Đã tải {y.size} ảnh từ {pi}")
        return x, y
    # b) file .txt
    pi, pl = find_file("test_images.txt"), find_file("test_labels.txt")
    if pi and pl:
        y = np.loadtxt(pl, dtype=np.int32).reshape(-1)
        x = _to_images(np.loadtxt(pi, dtype=np.float32, delimiter=",").reshape(-1), y.size)
        print(f"-> Đã tải {y.size} ảnh từ {pi}")
        return x, y
    # c) tập MNIST test có sẵn trong Keras (bạn đã cài khi train) rồi xuất .bin cho code C
    try:
        try:
            from tensorflow.keras.datasets import mnist
        except ImportError:
            from keras.datasets import mnist
        (_, _), (xt, yt) = mnist.load_data()
        x = (xt.astype(np.float32) / 255.0).reshape(-1, IMG, IMG)
        y = yt.astype(np.int32)
        print(f"-> Đã tải {y.size} ảnh MNIST test từ Keras")
        save_dir = next((d for d in SEARCH_DIRS if os.path.isdir(d)
                         and os.path.basename(d).lower() == "data"), WEIGHTS_DIR)
        os.makedirs(save_dir, exist_ok=True)
        x.tofile(os.path.join(save_dir, "test_images.bin"))
        y.tofile(os.path.join(save_dir, "test_labels.bin"))
        print(f"-> Đã xuất test_images.bin / test_labels.bin vào {save_dir} (cho code C dùng)")
        return x, y
    except Exception as e:
        raise FileNotFoundError(
            "Không tìm thấy test_images.* / test_labels.* và cũng không tải được MNIST từ Keras "
            f"({e}). Hãy đặt test_images.bin + test_labels.bin vào D:\\reports\\data."
        )


if __name__ == "__main__":
    params = load_model_params()
    x_raw, y_true = load_test_data()
    BATCH_SIZE = x_raw.shape[0]

    # Padding như lúc train (nếu mô hình được train với ảnh 30x30) - không tính vào thời gian
    pad = params["pad"]
    x_batch = np.pad(x_raw, ((0, 0), (pad, pad), (pad, pad))) if pad else x_raw
    x_batch = np.ascontiguousarray(x_batch, dtype=np.float32)

    # Chạy làm nóng (Warm-up)
    _ = infer_batch_python(x_batch[:min(10, BATCH_SIZE)], params)

    # --- BẮT ĐẦU ĐO THỜI GIAN SUY LUẬN ---
    start_time = time.perf_counter()
    logits_batch = run_chunked(x_batch, params)
    end_time = time.perf_counter()
    # --- KẾT THÚC ĐO THỜI GIAN ---

    predictions = np.argmax(logits_batch, axis=1)
    correct = int(np.sum(predictions == y_true))
    incorrect = BATCH_SIZE - correct
    accuracy = correct / BATCH_SIZE * 100
    total_time_ms = (end_time - start_time) * 1000

    print(f"\n========== BÁO CÁO PYTHON (NUMPY) ==========")
    print(f"Tổng số ảnh kiểm tra : {BATCH_SIZE} ảnh")
    print(f"Dự đoán ĐÚNG         : {correct} ảnh")
    print(f"Dự đoán SAI          : {incorrect} ảnh")
    print(f"Tỉ lệ chính xác      : {accuracy:.2f}%")
    print(f"--------------------------------------------")
    print(f"Tổng thời gian chạy  : {total_time_ms:.2f} ms")
    print(f"Thời gian/1 ảnh      : {total_time_ms / BATCH_SIZE:.5f} ms")
    print(f"============================================")