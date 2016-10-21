import numpy as np
from PIL import Image


def image_to_numpy(img):
    return np.array(img) / 255.0


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def image_to_FC1(img):
    return rgb2gray(image_to_numpy(img))


def image_to_invFC1(img):
    return 1.0 - rgb2gray(image_to_numpy(img))


def numpy_to_image(mat):
    return Image.fromarray(mat * 255.0, 'F')
