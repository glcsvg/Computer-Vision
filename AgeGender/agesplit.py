# Import required modules
import cv2 
import math
import time
import argparse
import glob
import shutil
import os

def getFaceBox(net, frame, conf_threshold=0.5):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            #cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using Opencv2.')
#parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument("--device", default="cpu", help="Device to inference on")

args = parser.parse_args()


args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

#ageProto = "age_deploy.prototxt"
#ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
#ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

out_image1 ='/home/dell/Desktop/GenderDetection/AgeGender/boy/'
out_image2 ='/home/dell/Desktop/GenderDetection/AgeGender/girl/'

if args.device == "cpu":
    #ageNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

    genderNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    
    faceNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

    print("Using CPU device")
elif args.device == "gpu":
    #ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")


# Open a video file or an image file or a camera stream
img_list = glob.glob('/home/dell/Desktop/GenderDetection/inface/*jpg')

#Loop Video Stream
for img in img_list:
    frame = cv2.imread(img)
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        #print(img)
        continue
    else:
        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]):min(bbox[3],frame.shape[0]-1),max(0,bbox[0]):min(bbox[2], frame.shape[1]-1)]
            if face is not None:
                print(face.shape[1],face.shape[0])
                if face.shape[1]==0 or face.shape[0]==0:
                    continue
                else:

                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    # print("Gender Output : {}".format(genderPreds))
                    print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
                    if gender == 'Male':
                        shutil.copy(img, out_image1)

                    else:
                        shutil.copy(img, out_image2)
                    #os.remove(img)
            else:
                continue

        # cv2.imwrite("age-gender-out-{}".format(args.input),frameFace)
    

 
# cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/opencv_gpu -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF -DOPENCV_ENABLE_NONFREE=ON -DOPENCV_EXTRA_MODULES_PATH=~/cv2_gpu/opencv_contrib/modules -DPYTHON_EXECUTABLE=~/env/bin/python3 -DBUILD_EXAMPLES=ON -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON  -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON  -DWITH_CUBLAS=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 -DOpenCL_LIBRARY=/usr/local/cuda-10.2/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda-10.2/include/ ..