import gen_small as gen
import network as ann


words, imgs = gen.get_tuples()
word_mat, img_mat = gen.prepare_input_tensors(words, imgs)

print "Input images >>", img_mat.shape
print "Input labels >>", word_mat.shape

nn = ann.Network([24, img_mat.shape[1], img_mat.shape[2]], word_mat.shape, gen.get_default_total_code(), 50)
nn.train(img_mat, word_mat, "test_weight", batch_size=24, max_iteration=1000, continue_from_last=False)
