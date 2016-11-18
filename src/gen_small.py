# -*- coding: utf-8 -*-

import itertools
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from random import randint
import util
import numpy as np

max_word_length_index = 0
max_word_length = 0

start_code = u"\u0E00"
end_code = u"\u0E07"


def indices_to_unicode(indices, start=start_code, end=end_code):
    out = u""
    for i in indices:
        out = out + unichr(ord(start_code) + i)
    return out + u" "


wordlist = []
all_indices = itertools.permutations([1, 2, 4, 7])
for indices in all_indices:
    word = indices_to_unicode(indices)
    print word
    wordlist.append(word)


for i in xrange(len(wordlist)):
    if len(wordlist[i]) > max_word_length:
        max_word_length = len(wordlist[i])
        max_word_length_index = i


print "total", len(wordlist), "words of maximum characters ", max_word_length
font = ImageFont.truetype("../fonts/THSarabunNew Bold.ttf", 24)


def get_max_word_length():
    return max_word_length


def get_default_total_code():
    return ord(end_code) - ord(start_code) + 1


def get_total_data():
    return len(wordlist)


def unicodes_to_indices(string_code, start=start_code, end=end_code):
    """convert unicodes to indices (optionally) within range [start,end] (inclusive)"""
    out = []
    for i in range(0, len(string_code)):
        out.append(ord(string_code[i]) - ord(start))
    return out


def random_tuple(index=randint(0, len(wordlist))):
    """get a (random if index is not given) tuple of (word, image, index)"""
    word = wordlist[index][:-1]
    img = Image.new("RGBA", (160, 30), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    ts = draw.textsize(u" " + word + u" ", font=font)
    # print ts
    draw.text(((160 - ts[0]) / 2, 0), u" " + word + u" ", (0, 0, 0), font=font)
    draw = ImageDraw.Draw(img)
    return (word, img, index)


def get_tuples(indices=range(len(wordlist))):
    """get lists of words, images"""
    words = []
    imgs = []
    for i in indices:
        w, m, o = random_tuple(index=i)
        words.append(w)
        imgs.append(m)
    return words, imgs


def prepare_input_tensors(words, imgs):
    """convert all to numpy array"""
    """unicode for thai starts at index 1 not 0"""
    m = np.ndarray([len(imgs), imgs[0].size[0], imgs[0].size[1]])
    for i in xrange(len(imgs)):
        m[i, ...] = np.transpose(util.image_to_invFC1(imgs[i]))
    w = np.zeros([len(words), get_max_word_length() + 1], dtype=np.int32)
    for i in xrange(len(words)):
        arr = np.array(unicodes_to_indices(words[i]))
        w[i, 0:arr.shape[0]] = arr
    return w, m


if __name__ == "__main__":
    word, img, index = random_tuple(max_word_length_index)
    img.show()
    img.save("../artifacts/test.png")
    print repr(word), index
    print unicodes_to_indices(word), get_default_total_code()
