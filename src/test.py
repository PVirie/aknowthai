import gen
import network as ann
import numpy as np
import util


def eval(neural_net, data, labels):
    classes, alphas = neural_net.scan(data, labels)
    util.plot_1D(alphas[0])

    # now get only classess corresponding to high alphas
    index_output = np.argmax(classes, axis=2)
    masked = np.reshape(alphas > 0.5, (alphas.shape[0], alphas.shape[1]))

    count = 0
    correct = 0
    for b in xrange(index_output.shape[0]):
        lc = 0
        for c in xrange(index_output.shape[1]):
            if(masked[b, c]):
                if(lc < labels.shape[1]):
                    correct += 1 if labels[b, lc] == index_output[b, c] else 0
                    lc += 1
                count += 1

    print "Percent correct = ", correct * 100.0 / count


words, imgs = gen.get_tuples(range(10))
word_mat, img_mat = gen.prepare_input_tensors(words, imgs)

nn = ann.Network(img_mat.shape, word_mat.shape, gen.get_default_total_code())
nn.load_session("test_weight")
eval(nn, img_mat, word_mat)
