from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

east = 'frozen_east_text_detection.pb'


def detect_text(image):
    
    # load the input image and grab the image dimensions
    
    orig = image.copy()
    (H, W) = image.shape[:2]
    print("eski",(H, W))

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (1024,1024)
    rW = W / float(newW)
    rH = H / float(newH)

    print("yeni",(newW, newH))
    print("oran",rH,rW)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.99:
                continue

            print("scores.............",scoresData[x])

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    
    if len(boxes) < 1 :
        print("text bulunmadı")
        return "text bulunmadı"
        
    else :
        boxes_sort = boxes[boxes[:,0].argsort()]
        #print("boxes.....",boxes_sort)

        new_list = []
        
        temp_indis = []
        for i in range(boxes_sort.shape[0]-1) :

            #print(boxes_sort[i],boxes_sort[i][2],boxes_sort[i+1][0],(boxes_sort[i+1][0]-boxes_sort[i][2]),(boxes_sort[i+1][1]-boxes_sort[i][3]))
        
            if ((boxes_sort[i+1][0]-boxes_sort[i][2]) < 11) and ((boxes_sort[i+1][1]-boxes_sort[i][3]) < 10 and (boxes_sort[i+1][1]-boxes_sort[i][3]) > -80):
                temp_indis = i + 1
                #print("temp",temp_indis)
                #print("burda")
                min_x = min(boxes_sort[i][0],boxes_sort[i][2],boxes_sort[i+1][0],boxes_sort[i+1][2])
                min_y = min(boxes_sort[i][1],boxes_sort[i][3],boxes_sort[i+1][1],boxes_sort[i+1][3])
                max_x = max(boxes_sort[i][0],boxes_sort[i][2],boxes_sort[i+1][0],boxes_sort[i+1][2])
                max_y = max(boxes_sort[i][1],boxes_sort[i][3],boxes_sort[i+1][1],boxes_sort[i+1][3])

                #print("yeni box",  [min_x,min_y,max_x,max_y])
                new_box = [min_x,min_y,max_x,max_y]
                new_list.append(new_box)
                
                
            else:
                
                if i == temp_indis:
                    #print("silinmeli",i)
                    continue
                    
                else:
                    #print("kalan indis",i)
                    new_list.append(boxes_sort[i]) 
                

                #temp_indis = [] """
        
        my_array = np.array(new_list)
        #print("son ",my_array)

        #görselleşme kısmı silinebilir
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in my_array:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW) 
            startY = int(startY * rH)
            endX = int(endX * rW) 
            endY = int(endY * rH) 
            #print(startX,startY,endX,endY)
            # draw the bounding box on the image
            name = 'out.jpg'
            #cv2.imwrite(name,orig[startY:endY,startX:endX])
            text = str(startX)
            cv2.putText(orig,text,(startX, startY) ,cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # show the output image
        cv2.imshow("Text Detection", orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return my_array