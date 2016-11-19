import gen_small as gen
import network as ann
import numpy as np
import util


def eval(neural_net, data, labels):
    classes, alphas = neural_net.scan(data, gen.get_default_total_code())

    data3ch = util.cvtColorGrey2RGB(data)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for b in xrange(alphas.shape[0]):
        for c in xrange(alphas.shape[1]):
            data3ch[b, c, int(np.floor((1.0 - alphas[b, c]) * data3ch.shape[2])), :] = red
    tile = util.make_tile(data3ch, rows=600, cols=800, flip=True)
    util.numpy_to_image(tile).show()

    util.save_txt(classes[0], "data.out")

    # now get only classess corresponding to high alphas
    index_output = np.argmax(classes, axis=2)

    count = 0
    correct = 0
    for b in xrange(labels.shape[0]):
        for c in xrange(labels.shape[1]):
            if labels[b, c] > 0:
                correct += 1 if labels[b, c] == index_output[b, c] else 0
                count += 1

    print "Percent correct = ", correct * 100.0 / count


words, imgs = gen.get_tuples(range(10))
word_mat, img_mat = gen.prepare_input_tensors(words, imgs)

nn = ann.Network(img_mat.shape, word_mat.shape, gen.get_default_total_code(), 50)
nn.load_session("test_weight")
eval(nn, img_mat, word_mat)

# raw_input()
