import gen
import network as ann
import numpy as np
import util


def eval(neural_net, data, labels):
    classes, alphas = neural_net.scan(data, gen.get_default_total_code())

    data3ch = util.cvtColorGrey2RGB(data)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for b in xrange(alphas.shape[0]):
        for c in xrange(alphas.shape[1]):
            data3ch[b, c, int(np.floor((1.0 - alphas[b, c]) * (data3ch.shape[2] - 1))), :] = red
    tile = util.make_tile(data3ch, rows=600, cols=800, flip=True)
    util.numpy_to_image(tile).show()

    # now get only classess corresponding to high alphas
    index_output = np.argmax(classes, axis=2)
    util.save_txt(index_output, "../artifacts/" + "data.out")

    count = 0
    correct = 0
    for b in xrange(labels.shape[0]):
        for c in xrange(labels.shape[1]):
            if labels[b, c] > 0:
                correct += 1 if labels[b, c] == index_output[b, c] else 0
                count += 1
    print "Percent correct = ", correct * 100.0 / count

    collector = []
    for b in xrange(alphas.shape[0]):
        read_index = 0
        converted = gen.indices_to_unicode(index_output[b])
        read_word = u""
        for c in xrange(alphas.shape[1]):
            if alphas[b, c] > 0.5:
                read_word = read_word + converted[read_index]
                read_index = read_index + 1
        print read_word
        collector.append(read_word)

    return collector


words, imgs = gen.get_tuples(range(100))
word_mat, img_mat = gen.prepare_input_tensors(words, imgs)

nn = ann.Network(img_mat.shape, word_mat.shape, gen.get_default_total_code(), 200)
nn.load_session("../artifacts/" + "test_weight")
eval(nn, img_mat, word_mat)

# raw_input()
