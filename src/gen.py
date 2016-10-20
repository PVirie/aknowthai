# -*- coding: utf-8 -*-

import os
import codecs
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from random import randint

wordlist = []
for file in os.listdir("./wordlist"):
    fullname = os.path.join("./wordlist", file)
    if file.endswith(".txt"):
        print "Reading:", fullname
        with codecs.open(fullname, encoding='utf-8') as f:
            wordlist = wordlist + f.readlines()

print "total", len(wordlist), "words"
font = ImageFont.truetype("fonts/THSarabunNew Bold.ttf", 36)


word = u" " + wordlist[randint(0, len(wordlist))][:-1] + u" "
img = Image.new("RGBA", (150, 40), (255, 255, 255))
draw = ImageDraw.Draw(img)
ts = draw.textsize(word, font=font)
print ts
draw.text(((150 - ts[0]) / 2, 0), word, (0, 0, 0), font=font)
draw = ImageDraw.Draw(img)
img.show()
# img.save("artifact/test.png")
print repr(word)
