# Hannah Gray
# 10/24/2025
# Version 1.0
# openCV Testing

import cv2
import numpy as np

#set HSV lower and upper bounds for blue
lowerBlue = np.array([80, 120, 20])
upperBlue = np.array([130, 255, 255])

#set HSV lower and upper bounds for red
lowerRed = np.array([170, 120, 20])
upperRed = np.array([180, 255, 255])

video = cv2.VideoCapture(0)

while True:
    success, img = video.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maskBlue = cv2.inRange(image, lowerBlue, upperBlue)
    maskRed = cv2.inRange(image, lowerRed, upperRed)

    #Box the blue objects on webcam
    contoursB, hierarchyB = cv2.findContours(maskBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contoursB) != 0:
        for contour in contoursB:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    
    #Box the red objects on webcam
    contoursR, hierarchyR = cv2.findContours(maskRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contoursR) != 0:
        for contour in contoursR:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("maskBlue", maskBlue)
    cv2.imshow("maskRed", maskRed)
    cv2.imshow("webcam", img)

    cv2.waitKey(1)
    
    #Enter q on keyboard to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()