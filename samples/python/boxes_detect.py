import numpy as np
import cv2 as cv

img =  cv.imread('../data/boxes.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

thresh_inv = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]

# Blur the image
blur = cv.GaussianBlur(thresh_inv,(1,1),0)

thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

# find contours
contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
iCount = 0
mask = np.ones(img.shape[:2], dtype="uint8") * 255
for c in contours:
    # get the bounding rect
    iCount += 1
    x, y, w, h = cv.boundingRect(c)
    if w*h>1000:
        cv.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -1)
        cv.putText(mask, str(iCount), (x+5, y+13), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # cv.putText(mask, 'Fedex', (x, y+10), cv.FONT_HERSHEY_SIMPLEX, 0.1, (255,255,255), 2)

res_final = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))

cv.imshow("boxes", mask)
txt = "final image "+str(iCount)
cv.imshow(txt, res_final)
cv.waitKey(0)
cv.destroyAllWindows()