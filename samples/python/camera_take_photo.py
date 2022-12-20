import cv2
camera = cv2.VideoCapture(0)
i = 0
while i < 1:
    #raw_input('Press Enter to capture')
    return_value, image = camera.read()
    cv2.imwrite('take_camera/opencv'+str(i)+'.png', image)
    i += 1
del(camera)