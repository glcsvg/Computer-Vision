from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
import MNN
import time
import os
import psutil

interpreter = MNN.Interpreter("ssrnet.mnn")
session = interpreter.createSession()
inputs = interpreter.getSessionInputAll(session)
outputs = interpreter.getSessionOutputAll(session)
input0 = inputs['input']
output0 = outputs['age']

imgs_path = '/home/dell/Desktop/age_gender_workspace/age-gender-test'
listdir = os.listdir(imgs_path)

for file_path in listdir:
    img_path = os.path.join(imgs_path, file_path)
    ori_image = cv2.imread(img_path)
    image = ori_image[..., ::-1]
    image = cv2.resize(image, (64, 64))
    #preprocess it
    image = (image - 123.) / 58
    #as cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)

    #model
    #create temporary tensors for copy in/out 
    time_time = time.time()
    tmp_input0 = MNN.Tensor((1, 3, 64, 64), MNN.Halide_Type_Float,image, MNN.Tensor_DimensionType_Caffe)
    #copy tensors values as inputs 
    input0.copyFrom(tmp_input0)
    interpreter.runSession(session)
    pred = interpreter.getSessionOutput(session).getData()

    print(pred)
print("inference time: {} s".format(round(time.time() - time_time, 4)))
print('RAM memory % used:', psutil.virtual_memory()[2])
# Getting usage of virtual_memory in GB ( 4th field)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)





   