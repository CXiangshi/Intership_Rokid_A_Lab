
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np


# In[3]:


tf.logging.set_verbosity(tf.logging.INFO)

def model(features, labels, mode):
    
    inputlayers = tf.reshape(features["x"], [-1, 28, 28 ,1], name='input')
    conv1 = tf.layers.conv2d(inputs=inputlayers, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name='conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name="conv2")
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2")
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64], name='flat')
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense")
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN, name="dropout")
    logits = tf.layers.dense(inputs=dropout, units=10, name='output')
    
    prediction = {
        "classes":tf.argmax(logits, axis=1),
        "probabilities":tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    evals_metric_ops = {
        "accuracy":tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=evals_metric_ops)
    


# In[4]:


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
traindatas = mnist.train.images
trainlabels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_datas = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


# In[ ]:


mnist_classifier = tf.estimator.Estimator(model_fn=model, model_dir="/Users/cxs/Downloads/test")
tensors_log = {"probabilities":"softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_log, every_n_iter=50)


# In[ ]:


train_input = tf.estimator.inputs.numpy_input_fn(x={"x":traindatas}, y=trainlabels, batch_size=200, num_epochs=None, shuffle=True)
mnist_classifier.train(input_fn=train_input, steps=2000, hooks=[logging_hook])

