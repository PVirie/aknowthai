# -*- coding: utf-8 -*-

import os
import codecs
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from random import randint

max_word_length_index = 0
max_word_length = 0
wordlist = []
for file in os.listdir("../wordlists"):
    fullname = os.path.join("../wordlists", file)
    if file.endswith(".txt"):
        print "Reading:", fullname
        with codecs.open(fullname, encoding='utf-8') as f:
            wordlist = wordlist + f.readlines()

for i in xrange(len(wordlist)):
    if len(wordlist[i]) > max_word_length:
        max_word_length = len(wordlist[i])
        max_word_length_index = i


print "total", len(wordlist), "words of maximum characters ", max_word_length
font = ImageFont.truetype("../fonts/THSarabunNew Bold.ttf", 24)

start_code = u"\u0E00"
end_code = u"\u0E7F"


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


def get_tuples(indices=range(100)):
    """get lists of words, images"""
    words = []
    imgs = []
    for i in indices:
        w, m, o = random_tuple(index=i)
        words.append(w)
        imgs.append(m)
    return words, imgs


if __name__ == "__main__":
    word, img, index = random_tuple(max_word_length_index)
    img.show()
    img.save("../artifacts/test.png")
    print repr(word), index
    print unicodes_to_indices(word), get_default_total_code()