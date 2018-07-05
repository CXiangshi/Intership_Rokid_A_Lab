import tensorflow as tf 
import numpy as np 
import os 

from tensorflow.examples.tutorials.mnist import input_data

log_path = '/Users/cxs/Downloads/log'

data_path = '/Users/cxs/Downloads/MNIST-data'


def train():
    
    #tf.logging.set_verbosity(tf.logging.INFO)
    
    mnist = input_data.read_data_sets(data_path)
    
    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')


    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)


    def cnn_layer(image_shaped_input):
        with tf.name_scope('conv1'):
            conv1 = tf.layers.conv2d(
                                inputs=image_shaped_input,
                                filters=32,
                                kernel_size=[5, 5],
                                padding="same",
                                activation=tf.nn.relu)
        with tf.name_scope('pool1'):
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2)

        with tf.name_scope('conv2'):
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
        
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

        return pool2_flat
    
    cnn_tensor = cnn_layer(image_shaped_input)

    print(cnn_layer)
    #fc_layer1 = nn_layer(cnn_layer,3136, 500, 'fc_layer1')

    dense = tf.layers.dense(cnn_tensor, 1024, activation=tf.nn.relu)
    
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(dense, keep_prob)
    
    y = tf.layers.dense(dropped, 10)


    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_path + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_path + '/test')
    tf.global_variables_initializer().run()
    

    def feed_dict(train):
        if train:
            xs, ys = mnist.train.next_batch(100)
            k = 0.6
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(10000):
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            if i % 100 == 99:
                #run_options = tf.RunOptions(TraceLevel=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                xs, ys = mnist.train.next_batch(100)
                k = 0.6
                summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys, keep_prob: k})
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()

train()
