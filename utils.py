import tensorflow as tf
import scipy.io
import numpy as np
import os
import wget


def get_svhn(base_dir='../../data/svhn/', split='train'):
    file = split + '_32x32.mat'
    
    os.makedirs(base_dir, exist_ok=True)
    if not os.path.exists(base_dir + file):
        print('Downloading the SVHN dataset into ' + base_dir + file)
        wget.download('http://ufldl.stanford.edu/housenumbers/' + file, out=base_dir + file)
        
    images = scipy.io.loadmat(base_dir + file)['X'].transpose((3, 0, 1, 2))
    
    return tf.data.Dataset.from_tensor_slices(images), images.shape[0]


def immerge(images, n_row=None, n_col=None):
    """Merge images to an image with (n_row * h) * (n_col * w).

    `images` is in shape of N * H * W(* C=1 or 3)
    """
    n = images.shape[0]
    if n_row:
        n_row = max(min(n_row, n), 1)
        n_col = int(n - 0.5) // n_row + 1
    elif n_col:
        n_col = max(min(n_col, n), 1)
        n_row = int(n - 0.5) // n_col + 1
    else:
        n_row = int(n ** 0.5)
        n_col = int(n - 0.5) // n_row + 1

    h, w = images.shape[1], images.shape[2]
    shape = (h * n_row,
             w * n_col)
    if images.ndim == 4:
        shape += (images.shape[3],)
    img = np.full(shape, 0, dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx % n_col
        j = idx // n_col
        img[j * (h):j * (h ) + h,
            i * (w ):i * (w ) + w, ...] = image

    return img
