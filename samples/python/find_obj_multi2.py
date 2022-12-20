# pip install imutils
# python.exe -m pip install --upgrade pip
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


def main():
    # Reading the image and the template
    #img = cv2.imread('pictures/capture_img.png')
    
    
    iLoop = 0
    while True:
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
