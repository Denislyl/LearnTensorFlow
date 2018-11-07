import pandas as pd
import numpy as np
import tensorflow as tf

# 1 加载数据集，把对输入和结果分开
train = pd.read_csv('DigitRecognizerTrain.csv')
images_train = train.iloc[:, 1:].values
labels_train = train.iloc[:, 0].values

test = pd.read_csv('DigitRecognizerTest.csv')
images_test = test.iloc[:, :].values

# 2 对输入进行处理
images_train = images_train.astype(np.float)
images_train = np.multiply(images_train, 1.0/255)
images_test = images_test.astype(np.float)
images_test = np.multiply(images_test, 1.0/255)

images_size = images_train.shape[1]

images_width = images_height = np.ceil(np.sqrt(images_size)).astype(np.uint8)

# 3 对结果进行处理
labels_count = np.unique(labels_train).shape[0]


# 进行one-hot编码
def dense_to_ont_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_ont_hot(labels_train, labels_count)
labels = labels.astype(np.uint8)

# 4 对训练集进行分批
batch_size = 64
n_batch = int(len(images_train)/batch_size)

# 5 创建一个简单的神经网络用来对图片进行识别
x = tf.placeholder('float', shape=[None, images_size])
y = tf.placeholder('float', shape=[None, labels_count])

weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))
result = tf.matmul(x,  weights) + biases
predictions = tf.nn.softmax(result)

# 6 创建损失函数，以交叉熵的平均值为衡量
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions)
    )

# 7 用梯度下降法优化参数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 8 初始化变量
init = tf.global_variables_initializer()

# 9 计算预测值
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 循环50轮
    for epoch in range(50):
        for batch in range(n_batch-1):
            batch_x = images_train[batch*batch_size:(batch+1)*batch_size]
            batch_y = labels[batch*batch_size:(batch+1)*batch_size]
            # 进行训练
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        batch_x = images_train[n_batch*batch_size:]
        batch_y = labels[n_batch*batch_size:]
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
    # 计算预测
    myPrediction = sess.run(predictions, feed_dict={x: images_test})

label_test = np.argmax(myPrediction, axis=1)
pd.DataFrame(label_test).to_csv('simpleNN1.csv')
