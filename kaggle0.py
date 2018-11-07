import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import os

# The competition datafiles are in the directory ../input
# Read competition data files:
filedir = os.listdir()
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

learning_rate = 0.01
training_iteration = 30
batch_size = 300
display_step = 2

trainfv = train.drop(['label'], axis=1).values.astype(dtype=np.float32)
trainLabels = train['label'].tolist()
ohtrainLabels = tf.one_hot(trainLabels, depth=10)
ohtrainLabelsNdarray = tf.Session().run(ohtrainLabels).astype(dtype=np.float64)
trainfv = np.multiply(trainfv, 1.0 / 255.0)

testData = test.values
testData = np.multiply(testData, 1.0 / 255.0)
# train_x,train_y = make_the_data_ready_conv(train_x,train_y)
# valid_x,valid_y = make_the_data_ready_conv(valid_x,valid_y)

x = tf.placeholder("float",[None,784])
y = tf.placeholder("float",[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
  model = tf.nn.softmax(tf.matmul(x,W) + b)
  
w_h = tf.summary.histogram("weights",W)
b_h = tf.summary.histogram("biases",b)

#loss function with maximum likelihood 
with tf.name_scope("cost_function") as scope:
  cost_function = -tf.reduce_sum(y * tf.log(model))
  tf.summary.scalar("cost_function",cost_function)

#optimization with SGD
with tf.name_scope("train") as scope:
  op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
init = tf.initialize_all_variables()
merged_summary_op = tf.summary.merge_all()

import math
from random import randint

#  for faster convergence
def random_batch(data,labels,size):
  value = math.floor(len(data) / size)    
  intervall = randint(0,value-1)
  return data[intervall*size:intervall*(size+1)],labels[intervall*size:intervall*(size+1)]

with tf.Session() as sess:
  sess.run(init)
  #run as single machine multiply GPU
  # view the stuff on tensorboard
  for iteration in range(training_iteration):
      avg_cost = 0
      total_batch = int(trainfv.shape[0]/batch_size)
      for i in range(total_batch):
          batch_xs,batch_ys = random_batch(trainfv,ohtrainLabelsNdarray,total_batch) 
          sess.run(op,feed_dict={x: batch_xs, y: batch_ys})
          avg_cost += sess.run(cost_function,feed_dict={x: batch_xs, y: batch_ys}) / total_batch
          if iteration % display_step == 0:
              print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))
  print( "Training Finished" )

  correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print( "\nAccuracy of the current model: ",sess.run(accuracy, feed_dict={x: trainfv[0:10000], y: ohtrainLabelsNdarray[0:10000]}))
  
  prob = sess.run(tf.argmax(model,1), feed_dict = {x: testData})
  which = 1
  print( 'predicted labe: {}'.format(str(prob[which])))
  
  print(prob)


writer = tf.summary.FileWriter("./TensorBoard/mnist_nn_log",sess.graph)
writer.close()
  #import csv
  #outputFile_dir = '../input/output.csv'
  #header = ['ImageID','Label']
  #with open(outputFile_dir, 'w', newline='') as csvFile:
  #    writer = csv.writer(csvFile, delimiter = ',')
  #    writer.writerow(header)
  #    for i, p in enumerate(prob):
  #        writer.writerow([str(i+1), str(p)])