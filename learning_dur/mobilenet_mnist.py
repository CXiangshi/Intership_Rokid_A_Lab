import tensorflow as tf 
import numpy as np 
import os 
import dw_pw_unit

from tensorflow.examples.tutorials.mnist import input_data

log_path = '/Users/cxs/Downloads/log'

data_path = '/Users/cxs/Downloads/MNIST-data'


def train():

    mnist = input_data.read_data_sets(data_path)

    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)


    def wieght_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def cnn_layer(image_shaped_input):


        def conv2d(x, W, dw=None):
            #if dw:
                #return tf.nn.depthwise_conv2d(x )
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='same')

        def max_pool(x):
            return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='same')

        W_conv1 = wieght_variable([5,5,1,1])
        b_conv1 = bias_variable([1])
        
        dw1_conv2d = tf.nn.relu(conv2d(image_shaped_input, W_conv1) + b_conv1)
        
        W_conv2 = wieght_variable([1,1,1,32])
        b_conv2 = bias_variable(32)

        pw1_conv2d = tf.nn.relu(conv2d(dw1_conv2d, W_conv2) + b_conv2)

        pool1 = max_pool(pw1_conv2d)

        W_conv3 = wieght_variable([5,5,32,32])
        b_conv3 = bias_variable([32])

        dw2_conv2d = tf.nn.relu(conv2d(pool1, W_conv3) + b_conv3)

        W_conv4 = wieght_variable([1,1,32,64])
        b_conv4 = bias_variable([64])

        pw2_conv2d = tf.nn.relu(conv2d(dw2_conv2d, W_conv4) + b_conv4)

        pool2 = max_pool(pw2_conv2d)
        
        pool_flat = tf.reshape(pool2, [-1,7*7*64])

        return pool_flat
    
    #cnn_tensor = cnn_layer(image_shaped_input)

    cnn_tensor = dw_pw_unit.depthwise_separable_conv(image_shaped_input, num_pwc_filters=32, width_multiplier=1, sc='dwconv1', downsample=True)
    cnn_tensor = dw_pw_unit.depthwise_separable_conv(cnn_tensor, num_pwc_filters=32, width_multiplier=1, sc='conv2', downsample=True)

    W_dense1 = wieght_variable([7*7*64, 1024])
    b_dense1 = bias_variable([1024])

    dense1 = tf.nn.relu(tf.matmul(cnn_tensor, W_dense1) + b_dense1)

    W_dense2 = wieght_variable([1024,10])
    b_dense2 = bias_variable([10])

    y = tf.nn.softmax(tf.matmul(dense1, W_dense2) + b_dense2)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)

    for i in range(1000):

            