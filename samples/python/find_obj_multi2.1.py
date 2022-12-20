# pip install imutils
# python.exe -m pip install --upgrade pip
import os
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

dir_path = 'img_compare'
img_count = 0
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        img_count += 1
        
def main():
    # Reading the image and the template
    #img = cv2.imread('pictures/capture_img.png')
    
    
    iLoop = 0
    while True:
        iCount = 0
        img = cv2.imread('pictures/capture_img.png')
        if iLoop == 0:
            iLoop = 1
            temp = cv2.imread('pictures/walnut.png')
        else:
            iLoop =0
            temp = cv2.imread('pictures/green_master.png')
        # save the image dimensions
        W, H = temp.shape[:2]
        
        # Define a minimum threshold
        thresh = 0.8
        
        # Converting them to grayscale
        img_gray = cv2.cvtColor(img,
        						cv2.COLOR_BGR2GRAY)
        temp_gray = cv2.cvtColor(temp,
        						cv2.COLOR_BGR2GRAY)                
        
        # Passing the image to matchTemplate method
        match = cv2.matchTemplate(
        	image=img_gray, templ=temp_gray,
        method=cv2.TM_CCOEFF_NORMED)
        
        # Select rectangles with
        # confidence greater than threshold
        (y_points, x_points) = np.where(match >= thresh)
        
        # initialize our list of rectangles
        boxes = list()
        
        # loop over the starting (x, y)-coordinates again
        for (x, y) in zip(x_points, y_points):
        	# update our list of rectangles
            boxes.append((x, y, x + W, y + H))
        boxes = non_max_suppression(np.array(boxes))
        
        # loop over the final bounding boxes
        for (x1, y1, x2, y2) in boxes:	
        	# draw the bounding box on the image
            iCount += 1
            cv2.putText(img, str(iCount), (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # Show the template and the final output
        #-->cv2.imshow("Template", temp)
        cv2.imshow("After NMS", img)
        cv2.waitKey(3000)
        cv2.destroyWindow("After NMS")
if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
