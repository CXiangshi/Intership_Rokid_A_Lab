
# coding: utf-8

# In[1]:
from models import TrainModel
import tensorflow as tf
import data_processing

TRAIN_TIMES = 30000
SHOW_STEP = 1
SAVE_STEP = 100

x_data = tf.placeholder(tf.int32, [64, None])
y_data = tf.placeholder(tf.int32, [64, None])
emb_keep = tf.placeholder(tf.float32)
rnn_keep = tf.placeholder(tf.float32)

data = data_processing.Dataset(64)

model = TrainModel(x_data, y_data, emb_keep, rnn_keep)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(TRAIN_TIMES):
        x, y = data.next_batch()
        loss, _ = sess.run([model.loss, model.optimize],
                            {model.data:x, model.labels:y, model.emb_keep:0.5 ,model.rnn_keep:0.5})
        if step % SHOW_STEP == 0:
            print('step {}, loss is {}'.format(step, loss))

        #if step % SAVE_STEP == 0:
            #saver.save(sess, '/Users/cxs/Downloads/heikeji', global_step=model.global_step())
