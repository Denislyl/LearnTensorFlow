import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_puntos = 2000
conjunto_puntos = []
for i in range(num_puntos):
    if np.random.random() > 0.5:
        conjunto_puntos.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

df = pd.DataFrame({"x": [v[0] for v in conjunto_puntos], "y": [v[1] for v in conjunto_puntos]})
sns.lmplot("x", "y", data=df, fit_reg=False, height=6)
plt.show()

vectors = tf.constant(conjunto_puntos)
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

print(vectors.get_shape())
print(centroides.get_shape())

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

print(expanded_vectors.get_shape())
print(expanded_centroides.get_shape())

diff = tf.subtract(expanded_vectors, expanded_centroides)
sqr = tf.square(diff)
distances = tf.reduce_sum(sqr, 2)
assignments = tf.argmin(distances, 0)

means = []
for c in range(k):
    means.append(tf.reduce_mean(tf.gather(vectors,tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), reduction_indices=[1]))
new_centroides = tf.concat(means,0)

update_centroides = tf.assign(centroides, new_centroides)

init_op = tf.global_variables_initializer()

num_steps = 100

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(num_steps):
        [_, centroid_values, points_values, assignment_values] = sess.run([update_centroides, centroides, vectors, assignments])
    print("centroids" + "\n", centroid_values)
plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
plt.show()
