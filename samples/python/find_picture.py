#! /usr/bin/python2.7
from PIL import Image, ImageDraw

im = Image.open ('pictures/zGjE6.png')
isize = im.size
walnut = Image.open ('pictures/walnut.png')
wsize = walnut.size
x0, y0 = wsize [0] // 2, wsize [1] // 2
pixel = walnut.getpixel ( (x0, y0) ) [:-1]

def diff (a, b):
    return sum ( (a - b) ** 2 for a, b in zip (a, b) )

best = (100000, 0, 0)
for x in range (isize [0] ):
    for y in range (isize [1] ):
        ipixel = im.getpixel ( (x, y) )
        d = diff (ipixel, pixel)
        if d < best [0]: best = (d, x, y)

draw = ImageDraw.Draw (im)
x, y = best [1:]
draw.rectangle ( (x - x0, y - y0, x + x0, y + y0), outline = 'red')
im.save ('out.png')