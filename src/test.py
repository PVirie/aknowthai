import gen_small as gen
import network as ann
import numpy as np
import util


def eval(neural_net, data, labels):
    classes, alphas = neural_net.scan(data)
    # util.plot_1D(alphas[5])
    # print alphas[5]

    # now get only classess corresponding to high alphas
    index_output = np.argmax(classes, axis=2)

    count = 0
    correct = 0
    for b in xrange(labels.shape[0]):
        for c in xrange(labels.shape[1]):
            correct += 1 if labels[b, c] == index_output[b, c] else 0
            count += 1

    print "Percent correct = ", correct * 100.0 / count


words, imgs = gen.get_tuples(range(10))
word_mat, img_mat = gen.prepare_input_tensors(words, imgs)

nn = ann.Network(img_mat.shape, word_mat.shape, gen.get_default_total_code(), 50)
nn.load_session("test_weight")
eval(nn, img_mat, word_mat)
