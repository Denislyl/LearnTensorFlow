from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from PIL import Image
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
slim = tf.contrib.slim


pb_file_path = 'frozen_deepID_v1_to_softmax.pb'
img_dir = '../single_face_test_data/real1.bmp'
img_dir_2 = '/home/weihua/models/research/slim/polar_skeleton_data/single_face_test_data/Capture_Aug_02_Color_1_crop.bmp'
label = 1
def image_tf_read(img_d):
    image_original = Image.open(img_d)
    pic = Image.merge('RGB', (image_original, image_original, image_original))
    image_arr = np.asarray(pic, dtype='float32')
    image = tf.image.resize_images(image_arr, [55, 47], method=0)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)
    sess_image = tf.Session()
    image = image.eval(session=sess_image)
    return image

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open(pb_file_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        logits = sess.graph.get_tensor_by_name("Softmax:0")
        img_tensor = sess.graph.get_tensor_by_name("Placeholder:0")

        image_1 = image_tf_read(img_dir)
        start_time = time.time()
        prediction = sess.run(logits, feed_dict={img_tensor: image_1})
        duration = time.time() - start_time
        print('The inference time for each image is %f' %duration)
        max_index = np.argmax(prediction, 1)
        print('预测的标签为：')
        print(max_index)
        print('预测的结果为：')
        print(prediction)

        if max_index == 0:
            print('This is a fake face with possibility %.6f' %prediction[:, 0])
        elif max_index == 1:
            print('This is a real face with possibility %.6f' %prediction[:, 1])



