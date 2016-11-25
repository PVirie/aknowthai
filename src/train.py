import gen
import network as ann


words, imgs = gen.get_tuples(range(100, 20100))
word_mat, img_mat = gen.prepare_input_tensors(words, imgs)

print "Input images >>", img_mat.shape
print "Input labels >>", word_mat.shape

nn = ann.Network([200, img_mat.shape[1], img_mat.shape[2]], word_mat.shape, gen.get_default_total_code(), 200)
nn.train(img_mat, word_mat, "../artifacts/" + "test_weight", batch_size=200, max_iteration=1000, continue_from_last=False)
