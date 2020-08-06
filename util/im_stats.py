import numpy as np

def mse(gnd_truth, im):
    assert gnd_truth.shape == im.shape
    gnd_truth = gnd_truth.ravel()
    im = im.ravel()
    return np.mean(np.power(np.subtract(gnd_truth, im),2))

def psnr(gnd_truth, im):
    assert gnd_truth.dtype.kind == im.dtype.kind and gnd_truth.shape == im.shape
    max_i = np.max(gnd_truth.ravel())
    return 20*np.log10(max_i) - 10*np.log10(mse(gnd_truth, im))