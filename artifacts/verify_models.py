"""Independent NumPy checks of original BN ordering and exported integer model."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

def load(name):
    return np.loadtxt(Path('weights') / (name+'.txt'), dtype=np.float32)

def conv(x,w):
    v=sliding_window_view(x,(3,3),axis=(1,2))
    v=v.transpose(0,1,2,4,5,3)
    return v.reshape(*v.shape[:3],-1) @ w

def pool(x):
    n,h,w,c=x.shape
    return x[:,:h//2*2,:w//2*2].reshape(n,h//2,2,w//2,2,c).max(axis=(2,4))

def float_reference(x,folded=False):
    for j,c in ((1,8),(2,16)):
        w=load(f'conv{j}_weights').reshape(-1,c)
        b=load(f'conv{j}_bias')
        a=load(f'bn{j}_gamma')/np.sqrt(load(f'bn{j}_var')+.001)
        t=load(f'bn{j}_beta')-a*load(f'bn{j}_mean')
        if folded:
            y=conv(x,w*a)+b*a+t
            y=np.where(a>=0,np.maximum(y,t),np.minimum(y,t))
        else:
            y=np.maximum(conv(x,w)+b,0)*a+t
        x=pool(y)
    return x.reshape(len(x),400)@load('dense_weights').reshape(400,10)+load('dense_bias')

def read_model(tag):
    with open(f'artifacts/{tag}_fpga.bin','rb') as f:
        assert f.read(4)==b'MNI8'
        assert np.fromfile(f,'<i4',1)[0]==1
        layers=[]
        for _ in range(3):
            m=np.fromfile(f,'<i4',8)
            _,ci,co,_,n,*_=m
            w=np.fromfile(f,'i1',n).astype(np.int32).reshape(-1,co)
            b,t,s=(np.fromfile(f,'<i4',co) for _ in range(3))
            layers.append((m,w,b,t,s))
        assert not f.read(1)
        return layers

def integer_reference(x,layers):
    x=np.clip(np.floor(np.ldexp(x.astype(np.float64),-int(layers[0][0][5]))+.5),-127,127).astype(np.int32)
    for m,w,b,t,s in layers[:2]:
        y=conv(x,w)+b
        y=np.where(s>0,np.maximum(y,t),np.minimum(y,t)).astype(np.int64)
        e=int(m[5]+m[6]-m[7])
        if e>=0:y=y*(1<<e)
        else:
            d=1<<(-e)
            y=np.sign(y)*((np.abs(y)+d//2)//d)
        x=pool(np.clip(y,-127,127).astype(np.int32))
    _,w,b,_,_=layers[2]
    return (x.reshape(len(x),400)@w+b).argmax(axis=1)

if __name__=='__main__':
    x=np.fromfile('data/test_images.bin','<f4').reshape(-1,28,28,1)
    y=np.fromfile('data/test_labels.bin','<i4')
    fp=[];maxerr=0
    for start in range(0,len(x),128):
        a=float_reference(x[start:start+128]);b=float_reference(x[start:start+128],True)
        maxerr=max(maxerr,float(np.max(np.abs(a-b))))
        assert np.allclose(a,b,atol=3e-5,rtol=1e-5)
        fp.extend(a.argmax(axis=1))
    pred=np.fromfile('artifacts/PTQ_predictions.bin','<i4').reshape(-1,2)
    assert np.array_equal(fp,pred[:,0])
    print(f'Original Conv-ReLU-BN vs folded: max logit error={maxerr:.9g}; C float predictions match 10000/10000; accuracy={np.mean(np.array(fp)==y)*100:.4f}%')
    for tag in ('PTQ','QAT'):
        layers=read_model(tag)
        pred=np.fromfile(f'artifacts/{tag}_predictions.bin','<i4').reshape(-1,2)
        q=integer_reference(x[:256],layers)
        assert np.array_equal(q,pred[:256,1])
        print(f'{tag}: exported INT8 independent prediction match 256/256; file bytes={Path(f"artifacts/{tag}_fpga.bin").stat().st_size}')
