"""Xuất PTQ và QAT từ đúng trọng số/kiến trúc Golden Model AI.c.

Chạy bằng ../.venv/Scripts/python.exe export_models.py từ thư mục này.
QAT dùng TensorFlow GradientTape và STE tùy chỉnh, không dùng TFLite converter.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.keras.utils.set_random_seed(2026)


def load(name, shape):
    # Chọn đúng tên file Golden Model đang đọc; không chọn các bản có dấu cách đầu tên.
    a = np.loadtxt(ROOT / 'weights' / (name + '.txt'), delimiter=',', dtype=np.float32)
    a = a.reshape(shape)
    if not np.isfinite(a).all():
        raise ValueError(name + ': NaN/Inf')
    return a


def golden_parameters():
    params = []
    for i, ci, co in [(1, 1, 8), (2, 8, 16)]:
        w = load(f'conv{i}_weights', (3, 3, ci, co))
        b = load(f'conv{i}_bias', (co,))
        gamma = load(f'bn{i}_gamma', (co,))
        beta = load(f'bn{i}_beta', (co,))
        mean = load(f'bn{i}_mean', (co,))
        var = load(f'bn{i}_var', (co,))
        # Golden là Conv -> BN -> ReLU: gộp BN trước khi lượng tử hóa.
        factor = gamma / np.sqrt(var + np.float32(1e-3))
        params += [w * factor, (b - mean) * factor + beta]
    params += [load('dense_weights', (400, 10)), load('dense_bias', (10,))]
    return params


def exponent(a):
    # Scale lũy thừa hai cho phép C requantize bằng nhân/chia số nguyên.
    return int(np.ceil(np.log2(max(float(np.max(np.abs(a))), 1e-12) / 127)))


def round_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quant(a, e, bias=False):
    q = round_away(np.asarray(a, np.float64) / (2.0 ** e))
    if bias:
        if np.max(np.abs(q)) > np.iinfo(np.int32).max:
            raise OverflowError('bias INT32')
        return q.astype(np.int32)
    return np.clip(q, -127, 127).astype(np.int8)


@tf.custom_gradient
def fake_quant(x, scale, limit):
    # Forward: lượng tử hóa rồi giải lượng tử; backward: STE trong miền biểu diễn.
    z = x / scale
    y = tf.clip_by_value(tf.sign(z) * tf.floor(tf.abs(z) + 0.5), -limit, limit) * scale

    def grad(dy):
        mask = tf.cast(tf.abs(z) <= limit, x.dtype)
        return dy * mask, None, None
    return y, grad


def forward(x, params, exps=None, intermediates=False):
    if exps is not None:
        x = fake_quant(x, float(2.0 ** exps['input']), 127.0)
    acts = []
    ei = exps['input'] if exps else None
    for j in range(3):
        w, b = params[2*j:2*j+2]
        if exps is not None:
            ew = exps['weights'][j]
            w = fake_quant(w, float(2.0 ** ew), 127.0)
            b = fake_quant(b, float(2.0 ** (ei + ew)), 2147483647.0)
        if j < 2:
            x = tf.nn.conv2d(x, w, strides=1, padding='VALID') + b
            x = tf.nn.relu(x)
            acts.append(x)
            if exps is not None:
                ei = exps['activations'][j]
                x = fake_quant(x, float(2.0 ** ei), 127.0)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
        else:
            # NHWC flatten khớp chính xác chỉ số trong Golden Model C.
            x = tf.reshape(x, [-1, 400]) @ w + b
    return (x, acts) if intermediates else x


def accuracy(x, y, params, exps=None):
    pred = []
    for start in range(0, len(x), 256):
        pred.extend(np.argmax(forward(x[start:start+256], params, exps).numpy(), axis=1))
    pred = np.asarray(pred, np.int32)
    return float(np.mean(pred == y)), pred


def export(tag, params, exps):
    dest = HERE / tag
    dest.mkdir(exist_ok=True)
    header = ['/* Trọng số xuất tự động; INT8 weights, INT32 bias, thứ tự HWIO/IO. */',
              '#include <stdint.h> /* Kiểu số nguyên có độ rộng cố định. */']
    ei = exps['input']
    layers = []
    for j in range(3):
        ew = exps['weights'][j]
        eo = exps['activations'][j] if j < 2 else ei + ew
        w = quant(np.asarray(params[2*j]), ew)
        b = quant(np.asarray(params[2*j+1]), ei + ew, True)
        # Chứng minh cả trường hợp INT8 xấu nhất vẫn không tràn tích lũy INT32.
        bound = np.abs(b.astype(np.int64)) + (w.size // len(b)) * 127 * 127
        assert np.max(bound) < 2147483647
        for name, array, ctype in [('w', w, 'int8_t'), ('b', b, 'int32_t')]:
            np.savetxt(dest / f'layer{j+1}_{name}.txt', array.reshape(-1), fmt='%d')
            array.astype('<i1' if name == 'w' else '<i4').tofile(dest / f'layer{j+1}_{name}.bin')
            header.append(f'static const {ctype} {name}{j+1}[{array.size}] = {{ /* Tham số lớp {j+1}. */')
            flat = array.reshape(-1)
            for start in range(0, len(flat), 16):
                header.append('    ' + ', '.join(map(str, flat[start:start+16])) + ', /* Các phần tử liên tiếp. */')
            header.append('}; /* Kết thúc mảng tham số. */')
        header.append(f'#define SHIFT{j+1} ({ei + ew - eo}) /* Số mũ đổi scale accumulator sang output. */')
        layers.append(dict(input_exp=ei, weight_exp=ew, output_exp=eo,
                           bias_exp=ei+ew, weight_shape=list(w.shape), bias_shape=list(b.shape)))
        ei = eo
    header.append(f'#define INPUT_EXP ({exps["input"]}) /* x_thực = x_INT8 * 2^INPUT_EXP. */')
    header.append(f'#define LOGIT_EXP ({ei}) /* logit_thực = accumulator * 2^LOGIT_EXP. */')
    (dest / 'weights.h').write_text('\n'.join(header) + '\n', encoding='utf-8')
    (dest / 'scales.json').write_text(json.dumps(dict(tag=tag, zero_point=0, qmin=-127, qmax=127,
             rounding='nearest, ties away from zero', layers=layers), indent=2), encoding='utf-8')
    np.savez(dest / 'float_checkpoint.npz', **{f'p{i}':np.asarray(p) for i,p in enumerate(params)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mnist', type=Path, default=Path.home()/'.keras/datasets/mnist.npz')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('QAT cần ít nhất một epoch huấn luyện')
    data = np.load(args.mnist)
    x = data['x_train'].astype(np.float32)[..., None] / 255.0
    y = data['y_train'].astype(np.int32)
    rng = np.random.default_rng(2026)
    indices = rng.permutation(len(x))
    train, val = indices[:55000], indices[55000:]
    params = [tf.constant(p) for p in golden_parameters()]
    _, acts = forward(x[train[:1024]], params, intermediates=True)
    exps = dict(input=exponent(np.array([1.0])),
                weights=[exponent(params[2*j]) for j in range(3)],
                activations=[exponent(a) for a in acts])
    export('PTQ', params, exps)
    print('PTQ exported; scales:', exps, flush=True)
    variables = [tf.Variable(p) for p in params]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def step(xb, yb):
        with tf.GradientTape() as tape:
            logits = forward(xb, variables, exps)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yb, logits=logits))
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    best = -1.0
    history = []
    for epoch in range(args.epochs):
        order = rng.permutation(train)
        losses = []
        for start in range(0, len(order), 128):
            batch = order[start:start+128]
            losses.append(float(step(x[batch], y[batch])))
            if start % (128*100) == 0:
                print(f'QAT epoch={epoch+1} samples={start}/{len(order)} loss={losses[-1]:.6f}', flush=True)
        acc, _ = accuracy(x[val], y[val], variables, exps)
        history.append(dict(epoch=epoch+1, mean_loss=float(np.mean(losses)), validation_accuracy=acc))
        print('QAT validation:', history[-1], flush=True)
        if acc > best:
            best = acc
            chosen = [tf.constant(v.numpy()) for v in variables]
    export('QAT', chosen, exps)
    xt = np.fromfile(ROOT/'data/test_images.bin', dtype='<f4').reshape(-1, 28, 28, 1)
    yt = np.fromfile(ROOT/'data/test_labels.bin', dtype='<i4')
    assert len(xt) == len(yt) == 10000
    results = dict(architecture='Conv-BN-ReLU-Pool-Conv-BN-ReLU-Pool-Dense (Golden C)',
                   tensorflow=tf.__version__, seed=2026, train_count=len(train), validation_count=len(val),
                   calibration_count=1024, qat_epochs=args.epochs, qat_history=history, scales=exps,
                   mnist_sha256=hashlib.sha256(args.mnist.read_bytes()).hexdigest(),
                   sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((ROOT/'weights').glob('*.txt')) if not p.name.startswith(' ')})
    for tag, p, e in [('Golden_float', params, None), ('PTQ', params, exps), ('QAT', chosen, exps)]:
        acc, pred = accuracy(xt, yt, p, e)
        results[tag] = dict(accuracy=acc, correct=int(np.sum(pred == yt)), count=len(yt))
        pred.astype('<i4').tofile(HERE/(tag+'_reference_predictions.bin'))
        print(tag, results[tag], flush=True)
    (HERE/'training_report.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
