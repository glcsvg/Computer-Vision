import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime  as nxrun
from onnx import numpy_helper
from skimage import io
from skimage.transform import resize
import glob
from PIL import Image,ImageDraw
from tensorflow.python.keras.preprocessing import image

onnx_model = "gender_ssr.onnx"
images = glob.glob('/home/dell/Desktop/age_gender_workspace/croped_face/*.jpg')

IMAGE_SIZE = 64


for i in images:
    #Preprocess the image
    # img = cv2.imread(image)
    # img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    # img = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_AREA)
    # img.resize((1, 112, 112, 3))
    
    # data = json.dumps({'data': img.tolist()})
    # data = np.array(json.loads(data)['data']).astype('float32')

    img = image.load_img(i, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)


    session = nxrun.InferenceSession(onnx_model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(session)
    print(input_name)
    print(output_name)
    feed = dict([(input.name, x[n]) for n, input in enumerate(session.get_inputs())])
    pred_onnx = session.run(None, feed)[0]
    pred = np.squeeze(pred_onnx)
    top_inds = pred.argsort()[::-1][:5]
    print(top_inds,pred)
        



























# images = glob.glob('/home/dell/Desktop/age_gender_workspace/croped_face/*.jpg')
# path='head_crop_99.jpg'

# for image in images:
#     #Preprocess the image
#     img = cv2.imread(image)
#     img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
#     img = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_AREA)
#     img.resize((1, 112, 112, 3))
    
#     data = json.dumps({'data': img.tolist()})
#     data = np.array(json.loads(data)['data']).astype('float32')
#     session = nxrun.InferenceSession(model, None)
#     input_name = session.get_inputs()[0].name
#     output_name = session.get_outputs()[0].name
#     print(session)
#     print(input_name)
#     print(output_name)
    
#     result = session.run([output_name], {input_name: data})
#     prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
#     print(result)
#     exit(1)
#     print("image {}: prediction {}".format(image,prediction))
























# img = Image.open('head_crop_99.jpg')
# img = img.resize((112, 112)) 
# X = np.asarray(img)
# X = X.transpose(2,0,1)
# X = X.reshape(1,112,112,3)


# x, y = img,1
# ort_sess = nxrun.InferenceSession('fashion_mnist_model.onnx')
# outputs = ort_sess.run(None, {'input': x.numpy()})

# # Print Result 
# predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
# print(f'Predicted: "{predicted}", Actual: "{actual}"')








# with open('labels.txt', 'r') as f:
#     labels = [l.rstrip() for l in f]




# from PIL import Image,ImageDraw
# sess = nxrun.InferenceSession(model)
# input_name = sess.get_inputs()[0].name
# img = Image.open('head_crop_99.jpg')
# img = img.resize((112, 112)) 
# X = np.asarray(img)
# X = X.transpose(2,0,1)
# X = X.reshape(1,112,112,3)
# out = sess.run(None, {input_name: X.astype(np.float32)})
# output_name = sess.get_outputs()[0].name
# print(sess)
# print(input_name)
# print(output_name)

# print(out)
# preds = np.squeeze(out)
# a = np.argsort(preds)[::-1]
# print(a)
# print('class=%s ; probability=%f' %(labels[a[0]],preds[a[0]]))


























# sess = nxrun.InferenceSession(model)

# y_pred = np.full(shape=(len(x_train)), fill_value=np.nan)

# for i in range(len(x_train)):
#     inputs = {}
#     for j in range(len(x_train.columns)):
#         inputs[x_train.columns[j]] = np.full(shape=(1,1), fill_value=x_train.iloc[i,j])

#     sess_pred = sess.run(None, inputs)
#     y_pred[i] = sess_pred[0][0][0]




# sess = nxrun.InferenceSession("./ssrnet_age-gender-onnx/ssrnet_3_3_3_112_1.0_1.0-gender.onnx")
# input_name = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name
# #pred = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]






# img = cv2.imread('head_crop_99.jpg')

# img = np.rollaxis(img, 2, 0) 
# img224 = resize(img / 255, (3, 112, 112), anti_aliasing=True)
# # ximg = img224[np.newaxis, :, :, :]
# # ximg = ximg.astype(np.float32)

# ximg = np.random.rand(1, 112, 112,3).astype(np.float32)


# print(type(ximg.shape))

# sess = nxrun.InferenceSession(model)
# print(sess.get_inputs()[0])

# print("The model expects input shape: ", sess.get_inputs()[0].shape)
# print("The shape of the Image is: ", ximg.shape)

# input_name = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name
# result = sess.run(None, {input_name: ximg})

# preds = np.squeeze(result)
# a = np.argsort(preds)[::-1]
#print('class=%s ; probability=%f' %(labels[a[0]],preds[a[0]]))






# prob = result[0]
# prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
# print(result,prediction,prob)
# #print("image {}: prediction {}".format(ximg,prediction))

# print(prob.ravel()[:10])