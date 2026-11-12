import os
import numpy as np

SRC_W = r"D:\reports\weights"
DST_W = r"D:\reports\weights_quantized"
DATA_DIR = r"D:\reports\data"
os.makedirs(DST_W, exist_ok=True)

# 1. Hàm lượng hóa Weights (Float -> INT8)
def quantize_and_save_w(name_in, name_out):
    w = np.loadtxt(os.path.join(SRC_W, name_in), delimiter=",")
    scale = 127.0 / max(np.max(np.abs(w)), 1e-7)
    qw = np.clip(np.round(w * scale), -128, 127).astype(np.int8)
    np.savetxt(os.path.join(DST_W, name_out), qw.flatten(), fmt="%d", delimiter=",")
    return scale

# 2. Hàm lượng hóa Bias (Float -> INT32)
def quantize_and_save_b(name_in, name_out, scale_w, scale_x=127.0):
    b = np.loadtxt(os.path.join(SRC_W, name_in), delimiter=",")
    qb = np.round(b * scale_w * scale_x).astype(np.int32)
    np.savetxt(os.path.join(DST_W, name_out), qb.flatten(), fmt="%d", delimiter=",")

print("Đang lượng hóa trọng số sang INT8...")
# Conv 1
s1 = quantize_and_save_w("conv1_weights.txt", "q_conv1_w.txt")
quantize_and_save_b("conv1_bias.txt", "q_conv1_b.txt", s1)
np.savetxt(os.path.join(DST_W, "q_bn1_mult.txt"), np.ones(8)*128, fmt="%d")
np.savetxt(os.path.join(DST_W, "q_bn1_shift.txt"), np.ones(8)*7, fmt="%d")

# Conv 2
s2 = quantize_and_save_w("conv2_weights.txt", "q_conv2_w.txt")
quantize_and_save_b("conv2_bias.txt", "q_conv2_b.txt", s2)
np.savetxt(os.path.join(DST_W, "q_bn2_mult.txt"), np.ones(16)*128, fmt="%d")
np.savetxt(os.path.join(DST_W, "q_bn2_shift.txt"), np.ones(16)*7, fmt="%d")

# Dense
sd = quantize_and_save_w("dense_weights.txt", "q_dense_w.txt")
quantize_and_save_b("dense_bias.txt", "q_dense_b.txt", sd)

# 3. Lượng hóa tập test images (Float32 -> INT8)
print("Đang lượng hóa ảnh test sang INT8...")
img_float = np.fromfile(os.path.join(DATA_DIR, "test_images.bin"), dtype=np.float32)
img_int8 = np.clip(np.round(img_float * 255.0 - 128.0), -128, 127).astype(np.int8)
img_int8.tofile(os.path.join(DATA_DIR, "test_images_int8.bin"))

print(f"Hoàn tất! Các file đã được xuất thành công tại {DST_W}")