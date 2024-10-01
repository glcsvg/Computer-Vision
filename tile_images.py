import numpy as np
import argparse
import time
import cv2
import os
import glob
import math
from demo_psp import segmente
from mxnet import image
from 
def pixCalc(y0, x0, box, height, width, horizon, lefBegin = 0, rightBegin = 0) :
    ymin = horizon + y0 + box[1]
    xmin = x0 + box[0] 
    #ymax = horizon + y0 + int(round((box[1]+box[3]) * height))
    #xmax = x0 + int(round((box[0]+box[2]) * width + rightBegin))
    return xmin, ymin

img_list = ['/home/visio-sevgi/Desktop/projects/signboardrules/m1.jpeg']

for i in img_list:

    #mage = cv2.imread(i)
    image = image.imread(i)
  
    #crop excesses ofimage   
    horizon = 1648
    bottomBegin = 1200

    print("padding ile resim",image.shape)
    
    numrows = 3
    numcols = 2
    intersection_size =100 #overlap size
    cwidth = image.shape[1]
    cheight = image.shape[0]
    height_for_each = int(cheight / numrows) #416
    width_for_each = int(cwidth / numcols)

    for row in range(numrows):
        for col in range(numcols):
            y0 = (row * height_for_each)
            x0 = (col * width_for_each)
            y1 = (y0 + height_for_each)
            x1 = (x0 + width_for_each)
            y1 = y1 + intersection_size
            x1 = x1 + intersection_size
            if row == 0 and col == 0:
                y0 = 0
                x0 = 0
                y1 = (y0 + height_for_each) + intersection_size
                x1 = (x0 + width_for_each) + intersection_size
            if y1 > cheight:
                y1 = cheight
                y0 = y0 - intersection_size
            if x1 > cwidth:
                x1 = cwidth
                x0 = x0 - intersection_size
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            crop_image = image[y0:y1, x0:x1]
            segmente(crop_image)
            
            """  print(crop_image.shape)
            cv2.imshow("Text Detection", crop_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
 """

