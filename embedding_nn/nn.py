import tensorflow as tf
import numpy as np
import random
import sklearn
from sklearn import preprocessing
from mnist import *

# ----------------------------------------------
#Adding seed for reproduction of results
np_rand = random.randint(0,10000)
from numpy.random import seed
seed(np_rand)

tf_rand = random.randint(0,10000)
from tensorflow import set_random_seed
set_random_seed(tf_rand)

print('np seed: ', np_rand)
print('tf seed: ', tf_rand)
# ----------------------------------------------

def one_hot(y_arr):
    one_hot_y = []
    for i in range(len(y_arr)):
        hot = np.zeros(10)
        hot[y_arr[i]] = 1
        one_hot_y.append(hot)

    return one_hot_y

print('Data snippit...')
# load same data as cvx fit
x_train, y_train, x_test, y_test = load()

# normalize input data
x_train = sklearn.preprocessing.normalize(x_train)
x_test = sklearn.preprocessing.normalize(x_test)

# convert y_train to tf usable shape (not (148,) np shape)
y_train = np.reshape(y_train, [len(y_train),1])
y_test = np.reshape(y_test, [len(y_test),1])

y_train = one_hot(y_train)
y_test = one_hot(y_test)

# number of variables we consider in each input
num_vars = len(x_train[0])

# after some searching, these seem to be optimal values
iter_ = 25000
lr = 1e-2
batch_size = 64
fc1_size = 32
fc2_size = 32
fc3_size = 32
fc4_size = 10
classes = len(y_train[0])

init = tf.initialize_all_variables()

# input placeholder
X = tf.placeholder(tf.float32, [None, num_vars])

# y_true
Y_ = tf.placeholder(tf.float32, [None, classes])

# fc1
W_1 = tf.Variable(tf.truncated_normal([num_vars, fc1_size]))
b_1 = tf.Variable(tf.truncated_normal([fc1_size]))
fc1 = tf.nn.softmax(tf.matmul(X, W_1) + b_1)

# fc2 
W_2 = tf.Variable(tf.truncated_normal([fc1_size, fc2_size]))
b_2 = tf.Variable(tf.truncated_normal([fc2_size]))
fc2 = tf.nn.softmax(tf.matmul(fc1, W_2) + b_2)

# fc3
W_3 = tf.Variable(tf.truncated_normal([fc2_size, fc3_size]))
b_3 = tf.Variable(tf.truncated_normal([fc3_size]))
fc3 = tf.matmul(fc2, W_3) + b_3

# fc4 
W_4 = tf.Variable(tf.truncated_normal([fc3_size, fc4_size]))
b_4 = tf.Variable(tf.truncated_normal([fc4_size]))
fc4 = tf.matmul(fc3, W_4) + b_4

# fc5 (out)
W_5 = tf.Variable(tf.truncated_normal([fc4_size, classes]))
b_5 = tf.Variable(tf.truncated_normal([classes]))
out = tf.matmul(fc4, W_5) + b_5

cost = tf.losses.mean_squared_error(Y_, out)
train = tf.train.GradientDescentOptimizer(lr).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(iter_):
        # sklearn shuffle to get minibatch
        shuffled_x, shuffled_y = sklearn.utils.shuffle(x_train, y_train, n_samples=batch_size)

        # train
        _, current_cost = sess.run([train, cost], feed_dict={X: shuffled_x,
                                                             Y_: shuffled_y})

        if i % 100 == 0:
            print('Iter: {0} Cost: {1}'.format(i, current_cost))

    print('\nTesting on test-set...')
    pred = sess.run(out, feed_dict={X: x_test})
    correct = 0
    
    for j in range(len(pred)):
        if np.argmax(pred[j]) == np.argmax(y_test[j]):
            correct += 1

    #print('\nTest batch size: ', len(y_test))
    print('Accuracy: {0}/{1}'.format(str(correct), str(len(y_test))))
