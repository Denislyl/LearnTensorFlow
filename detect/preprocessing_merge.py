from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import nets_factory
import numpy as np
from PIL import Image
import time
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

slim = tf.contrib.slim
offset = 150

log_dir = 'DeepIDv1_checkpoints/'
img_dir_1 = 'single_face_test_data/depth_1.png'   ### img_after
#img_dir_2 = 'single_face_test_data/fake7.bmp'
label = 1

# preprocess
def image_tf_read(input_img):

    # FIRST preprocess
    depth_input = cv2.imread(input_img, cv2.IMREAD_ANYDEPTH)
    ret, depth_inv = cv2.threshold(depth_input, 10 , 10000, cv2.THRESH_BINARY_INV)
    depth_out = depth_input + depth_inv
    depth_array = np.asarray(depth_out,  dtype='float32')
    
    min_val = np.min(depth_array)
    max_val = min_val + float(offset)

    for index_x, value_x in enumerate(depth_array):
        for index_y, value_y in enumerate(value_x):
            if value_y <= max_val:
                depth_array[index_x][index_y] = (depth_array[index_x][index_y] - min_val)/offset
            else:
                depth_array[index_x][index_y] = 1


    depth_new = depth_array * 255
    
    # SECOND preprocess ??????????

    depth_new = np.expand_dims(depth_new, 2)
    pic = np.concatenate((depth_new, depth_new, depth_new), axis=2)

    image = tf.image.resize_images(pic, [55, 47], method=0)
    image = tf.image.per_image_standardization(image)
   # image = tf.expand_dims(image, 0)
    sess_image = tf.Session()   
    image_after = image.eval(session=sess_image)
    return image_after

with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.train.get_or_create_global_step()

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        'deepID_v1',
        num_classes=2,
        is_training=False)
    img_tensor = tf.placeholder(tf.float32, shape=[1, 55, 47, 3])
    label = tf.cast(label, tf.int32)
    net6, _, duration_inference = network_fn(img_tensor)
    logits = tf.nn.softmax(net6)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess,  ckpt.model_checkpoint_path)
            print('Loading success')
        else:
            print('No checkpoint')
        
        # test 
        
        
        image_for_test = image_tf_read(img_dir_1)  ##### img_after  
        start_time = time.time()
        prediction = sess.run(logits, feed_dict={img_tensor: image_for_test})
        duration = time.time() - start_time
        print('The inference time for each image is %f' %duration)
        max_index = np.argmax(prediction)
        print('预测的标签为：')
        print(max_index)
        print('预测的结果为：')
        print(prediction)

        if max_index == 0:
            print('This is a fake face with possibility %.6f' %prediction[:, 0])
        elif max_index == 1:
            print('This is a real face with possibility %.6f' %prediction[:, 1])


