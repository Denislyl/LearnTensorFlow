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
import detect_face_pb
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709

saved_model_dir = "saved_model_dir_signature"

print('Creating networks and loading parameters')

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face_pb.create_mtcnn(sess, saved_model_dir)

image_path = 'I:/work/images'

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
save_dir = 'E:/MTCNN.txt'
f1 = open(save_dir, 'w')
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
'''
save_dir = 'E:/MTCNN.txt'
f1 = open(save_dir, 'w')
for file in files:
    if file.endswith('.jpg'):
        count += 1
        print('%d images'%count)
        img = misc.imread(file)
        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        points = points.T
        #print('Total face number：{}'.format(nrof_faces))
        #print(bounding_boxes)
        #print(points)
        #f1.write(file + ' %.2f %.2f %.2f %.2f'%(bounding_boxes[0][0], bounding_boxes[0][1], bounding_boxes[0][2], bounding_boxes[0][3]))

        imagewritepath = os.path.basename(file)
        if len(bounding_boxes) > 0:
            f1.write(imagewritepath + ' ')
        if len(bounding_boxes) > 1:
            f1.write('%d'%len(bounding_boxes))
            f1.write('\n')
        for i in range(len(bounding_boxes)):
            #f1.write('%d %d %d %d %f'%(int(bbox[i][0]), int(bbox[i][1]), int(bbox[i][2] - bbox[i][0]), int(bbox[i][3] - bbox[i][1]), bbox[i][4]))
            for j in range(5):
                f1.write('%f %f '%(points[i][j], points[i][j+5]))
            f1.write('\n')

f1.close()

'''
for file in files:
    count += 1
    print('%d images'%count)
    img = misc.imread(file)
    img1 = cv2.imread(file)
    bounding_boxes, points = detect_face_pb.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    print('Total face number?{}'.format(nrof_faces))
    
    crop_faces = []
    for i, face_position in enumerate(bounding_boxes):
        #face_position = face_position.astype(int)
        print(face_position[0:4])
        #cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        #cv2.putText(img1,str(np.round(face_position[4],2)),(int(face_position[0]),int(face_position[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        cv2.rectangle(img1, (int(face_position[0]), int(face_position[1])), (int(face_position[2]), int(face_position[3])), (0, 0, 255), 2)
        '''
        for j in range(5):
            #cv2.circle(img, (points[j][i], points[j+5][i]), 2, (255, 0, 0), 2)
            cv2.circle(img1, (points[j][i], points[j+5][i]), 2, (0,0,255))
        '''
        #crop = img[face_position[1]:face_position[3],
               #face_position[0]:face_position[2],]
    
        #crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
        #print(crop.shape)
        #crop_faces.append(crop)
        #plt.imshow(crop)
        #plt.show()
    
    '''
    plt.imshow(img)
    plt.show()
    '''
    '''
    cv2.imshow("lala",img1)
    cv2.waitKey(0)
    '''
    save_file = os.path.join(image_path, "%s.png"%count)
    cv2.imwrite(save_file,img1)
