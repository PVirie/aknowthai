import gen
import util
import numpy as np


def prepare_input_tensors(words, imgs):
    """convert all to numpy array"""
    w = np.zeros([gen.get_max_word_length(), len(words)])
    for i in xrange(len(words)):
        arr = np.array(gen.unicodes_to_indices(words[i]))
        w[0:arr.shape[0], i] = arr
    m = np.ndarray([imgs[0].size[1], imgs[0].size[0], len(imgs)])
    for i in xrange(len(imgs)):
        m[..., i] = util.image_to_invFC1(imgs[i])
    return w, m


w, m, i = gen.random_tuple()
print util.image_to_FC1(m).shape

words, imgs = gen.get_tuples(range(10))
prepare_input_tensors(words, imgs)
