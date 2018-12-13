import tensorflow as tf
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

tf.reset_default_graph()
sess = tf.Session()
x = tf.placeholder('float',  shape=[None, 28, 28, 1])
y = tf.placeholder('float',  shape=[None, 10])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

h_conv1 = tf.nn.conv2d(input=x, filter=W_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1
h_conv1 = tf.nn.relu(h_conv1)
h_pooll = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv2d(x, w):
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

h_conv2 = conv2d(h_pooll, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_ = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

crossEnteropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
trainStep = tf.train.AdamOptimizer().minimize(crossEnteropyLoss)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess.run(tf.global_variables_initializer())
batchSize = 128

m_saver = tf.train.Saver()

for i in range(10000):
    batch = mnist.train.next_batch(batchSize)
    trainingInputs = batch[0].reshape([batchSize, 28, 28, 1])
    labelsInputs = batch[1]
    trainStep.run(session=sess, feed_dict={x:trainingInputs, y:labelsInputs, keep_prob:0.5})
    if (i+1) %1000 == 0:
        testBatch = mnist.test.next_batch(batchSize)
        testInputs = testBatch[0].reshape([batchSize, 28, 28, 1])
        testLabelsInputs = testBatch[1]
        testaccuracy = accuracy.eval(session=sess, feed_dict={x:testInputs, y:testLabelsInputs, keep_prob:1})
        #testaccuracy = sess.run(accuracy, feed_dict={x:testInputs, y:testLabelsInputs, keep_drop:1})
        print("step:%d, test accuracy:%g"%(i, testaccuracy))
        m_saver.save(sess,  'I:/TF/model',  global_step=i)

writer = tf.summary.FileWriter("./mnist_nn_log",sess.graph)
writer.close()
