# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def deepID_v1_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope([slim.conv2d], padding='VALID'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc

def weight_variable(shape):
    with tf.name_scope('weights'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    with tf.name_scope('biases'):
        return tf.Variable(tf.zeros(shape))

def deepID_v1(inputs,
               num_classes=60,
               is_training=True,
               spatial_squeeze=False,
               scope='deepid_v1'):
  """AlexNet version 2.

  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224 or set
        global_pool=True. To use in fully convolutional mode, set
        spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: the number of predicted classes. If 0 or None, the logits layer
    is omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      logits. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original AlexNet.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0
      or None).
    end_points: a dict of tensors with intermediate activations.
  """
  start_inference = time.time()
  with tf.variable_scope(scope, 'deepID_v1', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      net1 = slim.conv2d(inputs, 20, [4, 4], 1, scope='conv1')
      net1 = slim.max_pool2d(net1, [2, 2], 2, scope='pool1')

      net2 = slim.conv2d(net1, 40, [3, 3], 1, scope='conv2')
      net2 = slim.max_pool2d(net2, [2, 2], 2, scope='pool2')

      net3 = slim.conv2d(net2, 60, [3, 3], 1, scope='conv3')
      net3 = slim.max_pool2d(net3, [2, 2], 2, scope='pool2')

      net4 = slim.conv2d(net3, 80, [2, 2], 1, scope='conv4')

    with tf.name_scope('fusion_feature'):
      h3r = tf.reshape(net3, [-1, 5 * 4 * 60])
      h4r = tf.reshape(net4, [-1, 4 * 3 * 80])

      W1 = weight_variable([5 * 4 * 60, 160])
      W2 = weight_variable([4 * 3 * 80, 160])
      b = bias_variable([160])

      h = tf.matmul(h3r, W1) + tf.matmul(h4r, W2) + b
      h5 = tf.nn.relu(h)

    # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(
        end_points_collection)

      if num_classes:

        net6 = slim.fully_connected(h5, num_classes, activation_fn=None, scope='fc8')

      if spatial_squeeze:
          net7 = tf.squeeze(net6, [1, 2], name='fc8/squeezed')
          end_points[sc.name + '/fc8'] = net7


      end_points[sc.name + '/fc8'] = net6
      duration_inference = time.time() - start_inference
      return net6, end_points, duration_inference

deepID_v1.default_image_size = 47
