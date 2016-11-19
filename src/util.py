import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

"""We use PIL for representing images."""


def image_to_numpy(img):
    return np.array(img) / 255.0


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def image_to_FC1(img):
    return rgb2gray(image_to_numpy(img))


def image_to_invFC1(img):
    return 1.0 - rgb2gray(image_to_numpy(img))


def numpy_to_image(mat):
    return Image.fromarray(np.uint8(mat * 255))


def plot_1D(mat):
    plt.plot(xrange(mat.shape[0]), mat, 'ro')
    plt.ion()
    plt.show()


def cvtColorGrey2RGB(mat):
    last_dim = len(mat.shape)
    return np.repeat(np.expand_dims(mat, last_dim), 3, last_dim)


def make_tile(mat, rows, cols, flip):
    b = mat.shape[0]
    r = mat.shape[2] if flip else mat.shape[1]
    c = mat.shape[1] if flip else mat.shape[2]
    canvas = np.zeros((rows, cols, 3 if len(mat.shape) > 3 else 1), dtype=mat.dtype)
    step = int(max(1, math.floor(b * (r * c) / (rows * cols))))
    i = 0
    for x in xrange(int(math.floor(rows / r))):
        for y in xrange(int(math.floor(cols / c))):
            canvas[(x * r):((x + 1) * r), (y * c):((y + 1) * c), :] = np.transpose(mat[i, ...], (1, 0, 2)) if flip else mat[i, ...]
            i = (i + step) % b

    return canvas


def save_txt(mat, name):
    np.savetxt("../artifacts/" + name, mat, delimiter=",", fmt="%.2e")
