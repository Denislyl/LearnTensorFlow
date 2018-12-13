from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from PIL import Image
import time
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from nets import nets_factory
slim = tf.contrib.slim

log_dir = 'DeepIDv1_checkpoints/'
img_dir_1 = 'single_face_test_data/real4.bmp'
img_dir_2 = 'single_face_test_data/fake7.bmp'
label = 1
def image_tf_read(img_dir):
    image_original = Image.open(img_dir)
    pic = Image.merge('RGB', (image_original, image_original, image_original))
    image_arr = np.asarray(pic, dtype='float32')
    
    image = tf.image.resize_images(image_arr, [55, 47], method=0)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)
    sess_image = tf.Session()
    image = image.eval(session=sess_image)

    return image

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
        image_1 = image_tf_read(img_dir_1)
        start_time = time.time()
        prediction = sess.run(logits, feed_dict={img_tensor: image_1})
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



