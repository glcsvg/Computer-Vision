import argparse as ap
import cv2
import imutils
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

train_path = args["trainingSet"]
training_names = os.listdir(train_path)  



image_paths = []  
image_classes = []  
class_id = 0
for training_name in training_names:  
    dir = os.path.join(train_path, training_name)
    class_path =  paths.list_images(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

fea_det = cv2.xfeatures2d.SIFT_create()

des_list = []
for image_path in image_paths:
    im = cv2.imread(image_path)
    des_ext = fea_det.detect(im, None)
    kpts, des = fea_det.compute(im, des_ext)
    des_list.append((image_path, des))  

descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

k = 500  
voc, variance = kmeans(descriptors, k, 1)  

im_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)  


np.savetxt("samples.data",im_features)
np.savetxt("responses.data",np.array(image_classes))
np.save("training_names.data",training_names)
np.save("stdSlr.data",stdSlr)
np.save("voc.data",voc)
