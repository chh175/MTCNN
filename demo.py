import tensorflow as tf
import numpy as np
import detect_face
from scipy import misc
import cv2
from matplotlib import pyplot as plt


minsize = 20
threshold = [0.6,0.7,0.7]
factor = 0.85

img = cv2.imread('3.jpg')
with tf.Session() as sess:
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)





#画图
for i, boxes in enumerate(bounding_boxes):
    
    box = boxes.astype(np.int32)
    landmark = points[:, i].reshape( (2,5) ).T
    
    draw_img = cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0,255,0), 2)
    for i, point in enumerate(landmark):
        if i==5:
            break
        draw_img = cv2.circle(draw_img,tuple(point),4,(255,0,0),-1)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',draw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()    


