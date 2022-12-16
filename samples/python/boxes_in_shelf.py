#!/usr/bin/env python

'''

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
from video import create_capture
from common import clock, draw_str

from datetime import datetime
log_file = open('log.txt', 'a')

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('polygon_box.jpg')))
    

    setCount = 0
    unixStamp = 0
    while True:
        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        thresh_inv = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
        blur = cv.GaussianBlur(thresh_inv,(1,1),0)
        thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        iCount = 0
        mask = np.ones(img.shape[:2], dtype="uint8") * 255
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv.boundingRect(c)
            if w*h>1000:
                iCount += 1
                cv.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -1)
                cv.putText(mask, str(iCount), (x+15, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        res_final = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))
        vis = res_final.copy()
        curUnix = datetime.now().timestamp()
        if (setCount != iCount) :
            if ((curUnix - unixStamp) > 10):
                unixStamp = curUnix
                setCount = iCount
                log_file.write('Counting : ' +  str(iCount) + ' Captutre time: ' + str(curUnix) + "\n")
                
        cv.imshow('box detect', vis)
        if cv.waitKey(5) == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    
    cv.destroyAllWindows()
