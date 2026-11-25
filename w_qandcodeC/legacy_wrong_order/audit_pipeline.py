"""Kiểm toán chỉ đọc mô hình/trọng số/dữ liệu; ghi kết quả vào audit/.

Không huấn luyện lại, không thay đổi Golden C, notebook hoặc trọng số hiện tại.
Chạy từ gốc dự án: .venv/Scripts/python.exe w_qandcodeC/audit_pipeline.py
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE/'audit'
OUT.mkdir(exist_ok=True)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)


def read(name, shape):
    return np.loadtxt(ROOT/'weights'/f'{name}.txt', delimiter=',', dtype=np.float32).reshape(shape)


def txt_params():
    p = []
    for i, ci, co in [(1, 1, 8), (2, 8, 16)]:
        p.append([read(f'conv{i}_weights', (3, 3, ci, co)), read(f'conv{i}_bias', (co,)),
                  *[read(f'bn{i}_{n}', (co,)) for n in ('gamma','beta','mean','var')]])
    p.append([read('dense_weights', (400, 10)), read('dense_bias', (10,))])
    return p


def txt_forward(x, params, order):
    # Cùng tensor, cùng dữ liệu, chỉ hoán đổi vị trí ReLU và BN để cô lập nguyên nhân.
    for w,b,g,beta,mean,var in params[:2]:
        x = tf.nn.conv2d(x, w, 1, 'VALID') + b
        if order == 'relu_bn':
            x = tf.nn.relu(x)
        x = g * (x - mean) / tf.sqrt(var + 0.001) + beta
        if order == 'bn_relu':
            x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, 2, 2, 'VALID')
    w,b = params[2]
    return tf.reshape(x, [-1, 400]) @ w + b


def evaluate(fn, x, y, filename):
    predictions = []
    for start in range(0, len(x), 256):
        scores = np.asarray(fn(x[start:start+256]))
        predictions.extend(np.argmax(scores, axis=1))
    predictions = np.asarray(predictions, dtype='<i4')
    predictions.tofile(OUT/filename)
    return dict(count=len(y), correct=int(np.sum(predictions == y)),
                accuracy=float(np.mean(predictions == y))), predictions


def run(command):
    p = subprocess.run([str(s) for s in command], cwd=ROOT, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.decode('utf-8', errors='replace')


def main():
    model = tf.keras.models.load_model(ROOT/'mnist_cnn_model.keras', compile=False)
    params = txt_params()
    x = np.fromfile(ROOT/'data/test_images.bin', dtype='<f4').reshape(-1, 28, 28, 1)
    y = np.fromfile(ROOT/'data/test_labels.bin', dtype='<i4')
    original = np.load(Path.home()/'.keras/datasets/mnist.npz')
    assert x.shape == (10000, 28, 28, 1) and y.shape == (10000,)
    result = dict(model_input_shape=list(model.input_shape),
                  dense_shape=list(model.layers[-1].get_weights()[0].shape),
                  images_equal_mnist=bool(np.array_equal(x[...,0], original['x_test'].astype(np.float32)/255)),
                  labels_equal_mnist=bool(np.array_equal(y, original['y_test'])),
                  test_images_int8_bytes=(ROOT/'data/test_images_int8.bin').stat().st_size,
                  expected_int8_bytes=10000*784,
                  model_layers=[dict(type=type(l).__name__, name=l.name,
                     activation=l.get_config().get('activation'), epsilon=l.get_config().get('epsilon')) for l in model.layers])
    comparisons = []
    for idx, names in [(0,['conv1_weights','conv1_bias']),
                       (1,[f'bn1_{n}' for n in ('gamma','beta','mean','var')]),
                       (3,['conv2_weights','conv2_bias']),
                       (4,[f'bn2_{n}' for n in ('gamma','beta','mean','var')]),
                       (8,['dense_weights','dense_bias'])]:
        for name, w in zip(names, model.layers[idx].get_weights()):
            txt = read(name, w.shape)
            entry = dict(name=name, count=w.size, shape=list(w.shape),
                         max_absolute_error=float(np.max(np.abs(txt.astype(np.float64)-w))),
                         equal_after_6_decimal_export=bool(np.array_equal(txt, np.array([float(f'{v:f}') for v in w.flat], dtype=np.float32).reshape(w.shape))))
            duplicate = ROOT/'weights'/(' '+name+'.txt')
            if duplicate.exists():
                a = np.loadtxt(duplicate, delimiter=',', dtype=np.float32).reshape(w.shape)
                entry['leading_space_duplicate_equal'] = bool(np.array_equal(txt,a))
            comparisons.append(entry)
    result['weight_comparisons'] = comparisons
    for tag, fn in [('keras_saved',lambda a:model(a,training=False)),
                    ('txt_relu_bn',lambda a:txt_forward(a,params,'relu_bn')),
                    ('txt_bn_relu',lambda a:txt_forward(a,params,'bn_relu'))]:
        result[tag], pred = evaluate(fn,x,y,tag+'_predictions.bin')
        print(tag,result[tag],flush=True)
    saved = np.fromfile(OUT/'keras_saved_predictions.bin',dtype='<i4')
    correct = np.fromfile(OUT/'txt_relu_bn_predictions.bin',dtype='<i4')
    wrong = np.fromfile(OUT/'txt_bn_relu_predictions.bin',dtype='<i4')
    result['keras_vs_correct_txt_prediction_mismatches'] = int(np.sum(saved != correct))
    result['correct_vs_wrong_order_prediction_mismatches'] = int(np.sum(correct != wrong))
    # So sánh xác suất và feature map của đúng thứ tự, không chỉ nhãn argmax.
    logits = txt_forward(x[:64],params,'relu_bn')
    result['first64_max_softmax_error'] = float(np.max(np.abs(model(x[:64],training=False).numpy()-tf.nn.softmax(logits).numpy())))
    # Xác nhận phép làm phẳng và kernel NumPy hiện tại cũng đúng trên toàn bộ tập test.
    spec = importlib.util.spec_from_file_location('existing_infer',ROOT/'notebooks/infer_python.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = module.load_model_params()
    result['numpy_existing'], np_pred = evaluate(lambda a:module.infer_batch_python(a[...,0],p),x,y,'numpy_predictions.bin')
    result['numpy_vs_keras_prediction_mismatches'] = int(np.sum(np_pred != saved))
    print('numpy_existing',result['numpy_existing'],flush=True)
    # Harness dùng nguyên hàm float Golden C, hai phiên bản chỉ khác thứ tự hai câu gọi.
    golden = (ROOT/'Source C/Golden Model AI.c').read_text(encoding='utf-8')
    source = '#define main golden_original_main\n' + golden + '\n#undef main\n#include <stdint.h>\n'
    source += 'int main(int argc,char **argv) {\n'
    shapes={'w1':72,'b1':8,'g1':8,'be1':8,'m1':8,'v1':8,'w2':1152,'b2':16,
            'g2':16,'be2':16,'m2':16,'v2':16,'wd':4000,'bd':10,'img':784,
            'c1':5408,'p1':1352,'c2':1936,'p2':400,'logits':10}
    for name,size in shapes.items():
        source+=f'static float {name}[{size}];\n'
    for j in (1,2):
        for prefix,fn in [('w','conv%d_weights'),('b','conv%d_bias'),('g','bn%d_gamma'),
                          ('be','bn%d_beta'),('m','bn%d_mean'),('v','bn%d_var')]:
            name=f'{prefix}{j}'
            source+=f'load_data("weights/{fn%j}.txt",{name},{shapes[name]});\n'
    source+='load_data("weights/dense_weights.txt",wd,4000);load_data("weights/dense_bias.txt",bd,10);\n'
    source+='FILE *f=fopen("data/test_images.bin","rb"),*o=argc==2?fopen(argv[1],"wb"):NULL; if(!f||!o)return 2;\n'
    source+='for(int n=0;n<10000;n++){if(fread(img,4,784,f)!=784)return 3;\n'
    for j,args,total,channels in [(1,'img,c1,w1,b1,28,28,1,8,3,3',5408,8),
                                 (2,'p1,c2,w2,b2,13,13,8,16,3,3',1936,16)]:
        source+=f'conv2d({args});\n'
        source+=f'BN_RELU_{j}\n'
        width=26 if j==1 else 11
        source+=f'max_pool2d(c{j},p{j},{width},{width},{channels});\n'
    source+='dense_layer(p2,wd,bd,logits,400,10);int32_t pred=argmax(logits,10);if(fwrite(&pred,4,1,o)!=1)return 4;}fclose(f);return fclose(o)!=0;}\n'
    with tempfile.TemporaryDirectory(dir=OUT) as temporary:
        directory=Path(temporary)
        for order in ('bn_relu','relu_bn'):
            s=source
            for j,total,ch in [(1,5408,8),(2,1936,16)]:
                bn=f'batch_norm_inplace(c{j},g{j},be{j},m{j},v{j},{total},{ch});'
                relu=f'relu_inplace(c{j},{total});'
                s=s.replace(f'BN_RELU_{j}',bn+relu if order=='bn_relu' else relu+bn)
            (directory/'check.c').write_text(s,encoding='utf-8')
            run(['gcc','-O3','-std=c11',directory/'check.c','-lm','-o',directory/'check.exe'])
            run([directory/'check.exe',OUT/f'golden_{order}_predictions.bin'])
            c=np.fromfile(OUT/f'golden_{order}_predictions.bin',dtype='<i4')
            ref=correct if order=='relu_bn' else wrong
            result['golden_C_'+order]=dict(correct=int(np.sum(c==y)),accuracy=float(np.mean(c==y)),
                                             reference_mismatches=int(np.sum(c!=ref)))
            assert np.array_equal(c,ref)
            print('golden_C_'+order,result['golden_C_'+order],flush=True)
    # Luôn đọc mã notebook hiện có; không dựa vào output lưu từ phiên chạy cũ.
    notebook=json.loads((ROOT/'notebooks/Training_step.ipynb').read_text(encoding='utf-8'))
    result['training_notebook_execution_counts']={str(i):c.get('execution_count') for i,c in enumerate(notebook['cells']) if c['cell_type']=='code'}
    result['quantized_existing']=json.loads((HERE/'verification_report.json').read_text(encoding='utf-8'))
    (OUT/'audit_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Results:',OUT/'audit_results.json',flush=True)


if __name__=='__main__':
    main()
