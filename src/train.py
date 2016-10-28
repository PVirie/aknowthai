import gen
import util
import numpy as np
import network as ann


def prepare_input_tensors(words, imgs):
    """convert all to numpy array"""
    m = np.ndarray([len(imgs), imgs[0].size[1], imgs[0].size[0]])
    for i in xrange(len(imgs)):
        m[i, ...] = util.image_to_invFC1(imgs[i])
    w = np.zeros([len(words), gen.get_max_word_length()], dtype=np.int32)
    for i in xrange(len(words)):
        arr = np.array(gen.unicodes_to_indices(words[i]))
        w[i, 0:arr.shape[0]] = arr
    return w, m


w, m, i = gen.random_tuple()
print util.image_to_FC1(m).shape

words, imgs = gen.get_tuples(range(1000))
word_mat, img_mat = prepare_input_tensors(words, imgs)

nn = ann.Network(img_mat.shape, word_mat.shape, gen.get_default_total_code())
nn.train(img_mat, word_mat, "test_weight", 100)
