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

    # cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalface_alt.xml")
    # nested_fn  = args.get('--nested-cascade', "haarcascades/haarcascade_eye.xml")
    # cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    # nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))
    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('osom.jpg')))
    
    while True:
        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        # rects = detect(gray, cascade)
        #img =  cv.imread('../data/boxes.png')
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
                # cv.putText(mask, 'img', (x, y+10), cv.FONT_HERSHEY_SIMPLEX, 0.1, (255,255,255), 2)

        res_final = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))

        #vis = img.copy()
        vis = res_final.copy()
        # draw_rects(vis, rects, (0, 255, 0))
        # if not nested.empty():
        #     for x1, y1, x2, y2 in rects:
        #         roi = gray[y1:y2, x1:x2]
        #         vis_roi = vis[y1:y2, x1:x2]
        #         subrects = detect(roi.copy(), nested)
        #         draw_rects(vis_roi, subrects, (255, 0, 0))
        #dt = clock() - t

        #draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        txt = str(iCount)
        cv.imshow('box detect', vis)
        #cv.imshow("boxes", mask)
        
        #txt = "final image "+str(iCount)
        #cv.imshow(txt, res_final)
        #cv.destroyAllWindows()
        if cv.waitKey(5) == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    
    cv.destroyAllWindows()
