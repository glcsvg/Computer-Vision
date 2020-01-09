import numpy as np
import cv2

trafic_sign_detec=cv2.CascadeClassifier("stopsign_classifier_haar.xml")

img = cv2.imread("images/1.jpg")
sign = trafic_sign_detec.detectMultiScale(img, 1.3, 5)
for(x,y,w,h) in sign:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    crop_img = img[y:y+h, x:x+w]
    filename = 'savedImage.jpg'
    cv2.imwrite(filename, crop_img) 

cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
