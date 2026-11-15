"""Export the cached MNIST training split, without network downloads."""
from pathlib import Path
import sys
import numpy as np

path = Path(sys.argv[1]) if len(sys.argv)>1 else Path.home()/'.keras/datasets/mnist.npz'
with np.load(path, allow_pickle=False) as d:
    assert d['x_train'].shape == (60000,28,28)
    assert d['y_train'].shape == (60000,)
    assert np.array_equal(d['y_test'],np.fromfile('data/test_labels.bin','<i4'))
    assert np.array_equal(d['x_test'].astype(np.float32)/255,
                          np.fromfile('data/test_images.bin','<f4').reshape(-1,28,28))
    (d['x_train'].astype('<f4')/255).tofile('artifacts/train_images.bin')
    d['y_train'].astype('<i4').tofile('artifacts/train_labels.bin')
print('Exported 60000 train images; test split matches existing project files.')
