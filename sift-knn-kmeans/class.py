import argparse as ap
import cv2
import imutils
import numpy as np
import os
from scipy.cluster.vq import *

samples = np.loadtxt('samples.data',np.float32)
responses = np.loadtxt('responses.data',np.float32)
classes_names = np.load('training_names.data.npy')
voc = np.load('voc.data.npy')
k = 500  

clf = cv2.KNearest()
clf.train(samples,responses)  

parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
args = vars(parser.parse_args())

image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print "No such directory {}\nCheck if the file exists".format(test_path)
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
else:
    image_paths = [args["image"]]
 
fea_det = cv2.xfeatures2d.SIFT_create()

des_list = []
for image_path in image_paths:
    im = cv2.imread(image_path)
    des_ext = fea_det.detect(im, None)
    kpts, des = fea_det.compute(im, des_ext)
    des_list.append((image_path, des)) 

descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))  

test_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1  

nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0) 
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

retval, results, neigh_resp, dists = clf.find_nearest(test_features,k=17)

if results[0][0] == 0:  
    prediction = "Traffic-Sign"
else:
    prediction = "Undefined"

if args["visualize"]:
    image = cv2.imread(image_path)
    cv2.namedWindow("Image",cv2.WINDOW_AUTOSIZE) 
    pt = (180,3*image.shape[0]//4)  
    cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2, [0, 0, 255],2) 
    cv2.imshow("Image",image)  
    cv2.waitKey()  
