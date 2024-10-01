import numpy as np
import argparse
import time
import cv2
import os
import glob
import math

net = cv2.dnn.readNet("/home/visio/darknet/traffic_v3/backup_traffic_v3/traffic_v3_30000.weights", "traffic_v3.cfg")
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)



LABELS = []
with open("traffic_v3.names", "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


img_list = glob.glob('/home/Desktop/tiles_image/*jpeg')

def pixCalc(y0, x0, box, height, width, horizon, lefBegin = 0, rightBegin = 0) :
    ymin = horizon + y0 + box[1]
    xmin = x0 + box[0] 
    #ymax = horizon + y0 + int(round((box[1]+box[3]) * height))
    #xmax = x0 + int(round((box[0]+box[2]) * width + rightBegin))
    return xmin, ymin

for i in img_list:

    IMAGE = cv2.imread(i)
    ac_height,ac_width = IMAGE.shape[0],IMAGE.shape[1]
    #crop excesses ofimage   
    horizon = 1848
    bottomBegin = 1224
    image = IMAGE[horizon:ac_height-bottomBegin,0:ac_width]
    #image = cv2.copyMakeBorder( image, 0, 0, 0, 148, 0, cv2.BORDER_CONSTANT, (255,0,0));
    print("padding ile resim",image.shape)
    
    numrows = 2
    numcols = 16
    intersection_size =100 #overlap size
    cwidth = image.shape[1]
    cheight = image.shape[0]
    height_for_each = int(image.shape[0] / numrows) #512
    width_for_each = int(image.shape[1] / numcols)
    print("width_for_each",width_for_each)
    print("height_for_each",height_for_each)

    for row in range(numrows):
        for col in range(numcols):
            y0 = (row * height_for_each)
            x0 = (col * width_for_each)
            y1 = (y0 + height_for_each)
            x1 = (x0 + width_for_each)
            y1 = y1 + intersection_size
            x1 = x1 + intersection_size
            if row == 0 and col == 0:
                y0 = 0;
                x0 = 0;
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
    
            #de
            start_time = time.time()
            traffic_signs = []
            idxs = []
            
            #image = cv2.resize(image, None, fx=0.5, fy=0.5)
            (H, W) = crop_image.shape[:2]
            print("crop_image shape",H,W)



            """ #blob1
            blob = cv2.dnn.blobFromImage(crop_image, 1 /300 , (H, W),swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            boxes = []
            confidences = []
            classIDs = []
            for output in layerOutputs:
                for detection in output:

                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > 0.0:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        traffic_signs.append((LABELS[classID],confidence))



            #visualization
            idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.1,0.5)


            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():

                    print("bu boxta",boxes[i])
                    # extract the bounding box coordinates
                    x,y = pixCalc(y0, x0, boxes[i], ac_height, ac_width, 1648,128)
                    
                    #(x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in COLORS[classIDS[i]]]
                    #print(x,y,w,h)
                    print("gercek box",x,y,x+w,y+h)
                    cv2.rectangle(IMAGE, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDS[i]], confidences[i])
                    cv2.putText(IMAGE, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)
                    #print(LABELS[classIDs[i]],confidences)
                    print(text)   
            else:
                print("no detect!", row,col)
                #continue
            print(traffic_signs)
            #cv2.imshow("Image", crop_416)
            #cv2.waitKey(0) """
            print("===========================================================================================================")
    cv2.imwrite("result.jpg",IMAGE)
                
