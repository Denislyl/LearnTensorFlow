# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 17:52:10 2018

@author: chenbihui
"""

'''
from six import string_types, iteritems
import numpy as np

data_dict = np.load('D:/Paper/facenet-master/src/align/det1.npy', encoding='latin1').item()
for op_name in data_dict:
    for param_name, data in iteritems(data_dict[op_name]):
        print(op_name, ':', param_name, data.shape)
'''

from scipy import misc
import tensorflow as tf
import detect_face
import cv2
import matplotlib.pyplot as plt
import os

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709

print('Creating networks and loading parameters')

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

image_path = 'D:/Paper/MTCNN-Tensorflow-master/test/lala'
save_dir = 'D:/Data/helen/trainset/300w_image.txt'
f1 = open(save_dir, 'w')

def list_all_files(rootdir):
    files = []
    file_list = os.listdir(rootdir)
    for i in range(len(file_list)):
        path = os.path.join(rootdir, file_list[i])
        if os.path.isdir(path):
            files.extend(list_all_files(path))
        if os.path.isfile(path):
            files.append(path)
    return files

files = list_all_files(image_path)
count = 0
'''
for file in files:
    if file.endswith('.jpg'):
        count += 1
        print('%d images'%count)
        img = misc.imread(file)
        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        #print('Total face number：{}'.format(nrof_faces))
        #print(bounding_boxes)
        #print(points)
        #f1.write(file + ' %.2f %.2f %.2f %.2f'%(bounding_boxes[0][0], bounding_boxes[0][1], bounding_boxes[0][2], bounding_boxes[0][3]))
        annotation_file = os.path.join(os.path.dirname(file), os.path.basename(file).replace('jpg', 'pts'))
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
            print(len(annotations))
            annotations = annotations[3:-1]
        landmarks = []
        for i in range(17, 68):
            annotation = annotations[i].strip().split(' ')
            annotation = list(map(float, annotation))
            landmarks.append(annotation)
        for i in range(nrof_faces):
            isbox = True
            for j in range(len(landmarks)):
                if landmarks[j][0] >= bounding_boxes[i][0] and landmarks[j][0] <= bounding_boxes[i][2] and landmarks[j][1] >= bounding_boxes[i][1] and landmarks[j][1] <= bounding_boxes[i][3]:
                    continue
                else:
                    isbox = False
                    break
            if isbox:
                f1.write(file + ' %.2f %.2f %.2f %.2f'%(bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][2], bounding_boxes[i][3]))
                for i in range(len(landmarks)):
                    f1.write(' %.3f %.3f'%(landmarks[i][0], landmarks[i][1]))
                f1.write('\n')

f1.close()
'''

for file in files:
    count += 1
    print('%d images'%count)
    img = misc.imread(file)
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    print('Total face number：{}'.format(nrof_faces))
    
    crop_faces = []
    for i, face_position in enumerate(bounding_boxes):
        face_position = face_position.astype(int)
        print(face_position[0:4])
        cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        for j in range(5):
            cv2.circle(img, (points[j][i], points[j+5][i]), 2, (255, 0, 0), 2)
        crop = img[face_position[1]:face_position[3],
               face_position[0]:face_position[2],]
    
        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
        print(crop.shape)
        crop_faces.append(crop)
        plt.imshow(crop)
        plt.show()
    
    plt.imshow(img)
    plt.show()