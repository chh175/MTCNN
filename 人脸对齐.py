import tensorflow as tf
import numpy as np
import detect_face
from scipy import misc
import cv2
from matplotlib import pyplot as plt

import face_preprocess


minsize = 20
threshold = [0.6,0.7,0.7]
factor = 0.85
image_size ='112,96'

#img = cv2.imread('2.jpg')
img = misc.imread('1.jpg')
with tf.Session() as sess:
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)


for i, boxes in enumerate(bounding_boxes):
    
    box = boxes.astype(np.int32)
    landmark = points[:, i].reshape( (2,5) ).T


    warped = face_preprocess.preprocess(img, bbox=box, landmark = landmark, image_size=image_size)

    cv2.namedWindow('image')
    cv2.imshow('image',warped[:,:,::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
