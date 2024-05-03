import cv2
import numpy as np
sr = cv2.dnn_superres.DnnSuperResImpl_create()

path = './models/LapSRN_x8.pb'
#path = './models/EDSR_x4.pb'
sr.readModel(path)
#sr.setModel('edsr',4) #lapsrn
sr.setModel('lapsrn',8) #lapsrn


sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


image = cv2.imread('img.jpg')

upscaled = sr.upsample(image)

cv2.imwrite('upscale.jpg',upscaled)

bicubic = cv2.resize(image,(upscaled.shape[1],upscaled.shape[0]),interpolation=cv2.INTER_CUBIC)
cv2.imwrite('cv2.jpg',bicubic)
