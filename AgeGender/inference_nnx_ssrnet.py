from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
import keras2onnx
import onnxruntime
import glob
import cv2
import time
import psutil


IMAGE_SIZE = 112
loop_count = 0
#img_path = 'r1.jpg'
#net = load_model('model-resnet50-final.h5')
#onnx_net = keras2onnx.convert_keras(net, net.name)
onnx_model = 'ssrnet_3_3_3_112_1.0_1.0-gender.onnx'


# 41 + 1 classes
cls_list = ['male', 'female']

images = glob.glob('/home/dell/Desktop/age_gender_workspace/croped_face/*.jpg')
path='head_crop_99.jpg'

start_time = time.time()

for img_path in images:
    #Preprocess the image

    # image preprocessing
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # runtime prediction
    #content = onnx_net.SerializeToString()
    #sess = onnxruntime.InferenceSession(content)
    sess = onnxruntime.InferenceSession(onnx_model)
    x = x if isinstance(x, list) else [x]
    feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
    pred_onnx = sess.run(None, feed)[0]
    pred = np.squeeze(pred_onnx)
    top_inds = pred.argsort()[::-1][:5]
    print(top_inds,pred)
    loop_count +=1
    # for i in top_inds:
    #     print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
print("ONNX inferences with %s second in average" %((time.time() - start_time) / loop_count))
# Getting % usage of virtual_memory ( 3rd field)
print('RAM memory % used:', psutil.virtual_memory()[2])
# Getting usage of virtual_memory in GB ( 4th field)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
