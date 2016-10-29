import gen
import network as ann

words, imgs = gen.get_tuples(range(10))
word_mat, img_mat = gen.prepare_input_tensors(words, imgs)

nn = ann.Network(img_mat.shape, word_mat.shape, gen.get_default_total_code())
nn.load_session("test_weight")
nn.eval(img_mat, word_mat)
