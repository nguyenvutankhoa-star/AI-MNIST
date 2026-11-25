"""Biên dịch, chạy 10000 ảnh và đối chiếu C với tham chiếu fake-quant TensorFlow.

Đồng thời biên dịch một harness dùng nguyên hàm của Golden Model để xác nhận baseline.
Không huấn luyện, không thay đổi trọng số. Chạy từ bất kỳ thư mục nào.
"""
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(command):
    completed = subprocess.run([str(s) for s in command], cwd=ROOT, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return completed.stdout.decode('utf-8', errors='replace')


def main():
    labels = np.fromfile(ROOT/'data/test_labels.bin', dtype='<i4')
    assert labels.shape == (10000,)
    result = {}
    for tag in ('PTQ', 'QAT'):
        exe = HERE/f'mnist_{tag}.exe'
        run(['gcc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror',
             HERE/f'mnist_{tag}.c', '-lm', '-o', exe])
        log = run([exe, '--test', ROOT/'data', HERE/f'{tag}_c_predictions.bin'])
        c = np.fromfile(HERE/f'{tag}_c_predictions.bin', dtype='<i4')
        ref = np.fromfile(HERE/f'{tag}_reference_predictions.bin', dtype='<i4')
        assert c.shape == ref.shape == (10000,)
        differences = int(np.sum(c != ref))
        assert differences == 0, f'{tag}: {differences} predictions differ from reference'
        # Chạy ảnh TXT mặc định từ cả gốc dự án và thư mục đầu ra.
        single = run([exe])
        local = subprocess.run([str(exe)], cwd=HERE, check=True, capture_output=True)
        assert single.encode() == local.stdout
        # Kiểm tra đường đọc ảnh không chấp nhận ảnh thiếu, thừa hoặc NaN.
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            image = Path(temporary)/'invalid.txt'
            for content in ('0\n'*783, '0\n'*785, 'nan\n'+'0\n'*783):
                image.write_text(content, encoding='ascii')
                bad = subprocess.run([str(exe), str(image)], cwd=ROOT, capture_output=True)
                assert bad.returncode != 0
        result[tag] = dict(correct=int(np.sum(c == labels)), count=10000,
                           accuracy=float(np.mean(c == labels)), reference_mismatches=differences,
                           strict_gcc_build='passed', invalid_input_checks='passed')
        (HERE/f'{tag}_test_results.txt').write_text(log + single, encoding='utf-8')
        print(tag, result[tag], flush=True)

    # Hàm main gốc được đổi tên để tái sử dụng nguyên các operator float, không sửa file nguồn.
    source = '#define main golden_original_main\n'
    source += '#include "' + (ROOT/'Source C/Golden Model AI.c').as_posix() + '"\n'
    source += '#undef main\n#include <stdint.h>\nint main(int argc, char **argv) {\n'
    shapes = {'w1':72, 'b1':8, 'gamma1':8, 'beta1':8, 'mean1':8, 'var1':8,
              'w2':1152, 'b2':16, 'gamma2':16, 'beta2':16, 'mean2':16, 'var2':16,
              'wd':4000, 'bd':10, 'img':784, 'c1':5408, 'p1':1352, 'c2':1936, 'p2':400, 'out':10}
    for name, count in shapes.items():
        source += f'static float {name}[{count}];\n'
    for i in (1, 2):
        mapping = {f'w{i}':f'conv{i}_weights', f'b{i}':f'conv{i}_bias',
                   **{f'{n}{i}':f'bn{i}_{n}' for n in ('gamma','beta','mean','var')}}
        for name, filename in mapping.items():
            source += f'load_data("weights/{filename}.txt", {name}, {shapes[name]});\n'
    source += 'load_data("weights/dense_weights.txt",wd,4000); load_data("weights/dense_bias.txt",bd,10);\n'
    source += '''FILE *f=fopen("data/test_images.bin","rb");
FILE *o=argc==2?fopen(argv[1],"wb"):NULL;
if(!f||!o) return 2;
for(int n=0;n<10000;n++) {
if(fread(img,sizeof(float),784,f)!=784) return 3;
conv2d(img,c1,w1,b1,28,28,1,8,3,3);
batch_norm_inplace(c1,gamma1,beta1,mean1,var1,5408,8);
relu_inplace(c1,5408); max_pool2d(c1,p1,26,26,8);
conv2d(p1,c2,w2,b2,13,13,8,16,3,3);
batch_norm_inplace(c2,gamma2,beta2,mean2,var2,1936,16);
relu_inplace(c2,1936); max_pool2d(c2,p2,11,11,16);
dense_layer(p2,wd,bd,out,400,10);
int32_t prediction=argmax(out,10);
if(fwrite(&prediction,4,1,o)!=1) return 4;
} fclose(f); return fclose(o)!=0; }
'''
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        path = Path(temporary)
        (path/'golden.c').write_text(source, encoding='utf-8')
        run(['gcc', '-O3', '-std=c11', path/'golden.c', '-lm', '-o', path/'golden.exe'])
        run([path/'golden.exe', path/'predictions.bin'])
        actual = np.fromfile(path/'predictions.bin', dtype='<i4')
    reference = np.fromfile(HERE/'Golden_float_reference_predictions.bin', dtype='<i4')
    assert actual.shape == reference.shape == (10000,)
    assert np.array_equal(actual, reference), 'Golden C differs from folded float reference'
    result['Golden_C'] = dict(correct=int(np.sum(actual == labels)), count=10000,
                               accuracy=float(np.mean(actual == labels)), folded_float_mismatches=0)
    print('Golden_C', result['Golden_C'], flush=True)
    # QAT phải thực sự thay đổi trọng số so với PTQ.
    for j in range(1, 4):
        ptq = np.loadtxt(HERE/'PTQ'/f'layer{j}_w.txt', dtype=np.int32)
        qat = np.loadtxt(HERE/'QAT'/f'layer{j}_w.txt', dtype=np.int32)
        result[f'layer{j}_changed_weights'] = int(np.sum(ptq != qat))
    assert sum(result[f'layer{j}_changed_weights'] for j in range(1, 4)) > 0
    (HERE/'verification_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
