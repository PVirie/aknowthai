import gen
import util
import numpy as np
import network as nn


def prepare_input_tensors(words, imgs):
    """convert all to numpy array"""
    m = np.ndarray([len(imgs), imgs[0].size[0], imgs[0].size[1]])
    for i in xrange(len(imgs)):
        m[i, ...] = np.transpose(util.image_to_invFC1(imgs[i]))
    w = np.zeros([len(words), imgs[0].size[1]], dtype=np.int32)
    for i in xrange(len(words)):
        arr = np.array(gen.unicodes_to_indices(words[i]))
        w[i, 0:arr.shape[0]] = arr
    return w, m


w, m, i = gen.random_tuple()
print util.image_to_FC1(m).shape

words, imgs = gen.get_tuples(range(1000))
word_mat, img_mat = prepare_input_tensors(words, imgs)

nn.train(img_mat, word_mat, gen.get_default_total_code(), "test_weight")
