import gen
import util
import numpy as np


def prepare_input_tensor(imgs):
    out = np.ndarray([len(imgs), imgs[0].size[0], imgs[0].size[1]])
    for i in xrange(len(imgs)):
        out[i, ...] = np.transpose(util.image_to_invFC1(imgs[i]))
    return out


w, m, i = gen.random_tuple()
print util.image_to_FC1(m).shape

words, imgs = gen.get_tuples(range(10))
prepare_input_tensor(imgs)
