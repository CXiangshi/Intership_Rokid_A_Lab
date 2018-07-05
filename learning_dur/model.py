import tensorflow as tf
from tensorflow.python.ops.gen_audio_ops import *
import numpy as np

import collections

poetry_file ='poetry.txt'  

poetrys = []  
with open(poetry_file, "r") as f:  
    for line in f:  
        try:  
            line = line.strip(u'\n')
            title, content = line.strip(u' ').split(u':')  
            content = content.replace(u' ',u'')  
            if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content:  
                continue  
            if len(content) < 5 or len(content) > 79:  
                continue  
            content = u'[' + content + u']'  
            poetrys.append(content)  
        except Exception as e:   
            pass  

poetrys = sorted(poetrys,key=lambda line: len(line))  
print('唐诗总数: ', len(poetrys))  

all_words = []  
for poetry in poetrys:  
    all_words += [word for word in poetry]  
counter = collections.Counter(all_words)  
count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
words, _ = zip(*count_pairs)  
 
words = words[:len(words)] + (' ',)  
 
word_num_map = dict(zip(words, range(len(words))))  

to_num = lambda word: word_num_map.get(word, len(words))  
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys] 

batch_size = 64
n_chunk = len(poetrys_vector) // batch_size 

class DataSet(object):
    def __init__(self,data_size):
        self._data_size = data_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(data_size)

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start + batch_size > self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed = self._epochs_completed + 1
            self._index_in_epoch = batch_size
            full_batch_features ,full_batch_labels = self.data_batch(0,batch_size)
            return full_batch_features ,full_batch_labels 
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            full_batch_features ,full_batch_labels = self.data_batch(start,end)
            if self._index_in_epoch == self._data_size:
                self._index_in_epoch = 0
                self._epochs_completed = self._epochs_completed + 1
                np.random.shuffle(self._data_index)
            return full_batch_features,full_batch_labels
 
    def data_batch(self,start,end):
        batches = []
        for i in range(start,end):
            batches.append(poetrys_vector[self._data_index[i]])

        length = max(map(len,batches))

        xdata = np.full((end - start,length), word_num_map[' '], np.int32)  
        for row in range(end - start):  
            xdata[row,:len(batches[row])] = batches[row]  
        ydata = np.copy(xdata)  
        ydata[:,:-1] = xdata[:,1:]  
        return xdata,ydata


def load_model(sess, saver, ckpt_path):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
        print ('resume from', latest_ckpt)
        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print ('building model from scratch')
        sess.run(tf.global_variables_initializer())
        return -1

def train_model():
    #tf.logging.set_verbosity(tf.logging.INFO)
    trainds = DataSet(len(poetrys_vector))
    sess = tf.InteractiveSession()

    rnn_size = 128
    nums_layers = 2

    with tf.name_scope('input'):
        input_data = tf.placeholder(tf.int32, [64, None])
        output_targets = tf.placeholder(tf.int32, [64, None])

    #logits, last_state, _, _, _ = neural_network()  
    targets = tf.reshape(output_targets, [-1])
  
      
    with tf.name_scope('rnn_lstm'):
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell
        cell = cell_fun(rnn_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*nums_layers, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):
        softmax_weights = tf.get_variable('softmax_w', [rnn_size, len(words)])
        softmax_bias = tf.get_variable('softmax_b', [len(words)])
        
        embedding = tf.get_variable('embedding', [len(words), rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
        output = tf.reshape(outputs, [-1, rnn_size])
    
        logits = tf.matmul(output, softmax_weights)+softmax_bias
        probs = tf.nn.softmax(logits)


    with tf.name_scope('cost'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=targets, logits=logits)
        cost = tf.reduce_mean(cross_entropy_mean)
    tf.summary.scalar('cost', cost)
    
    with tf.name_scope('learning_rate'):    
        learning_rate = tf.Variable(0.0, trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('train'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate)   
        train_op = optimizer.apply_gradients(zip(grads, tvars)) 
        
    init = tf.initialize_all_variables()
    sess.run(init)
        
    tf.global_variables_initializer()
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/home/ccc/code/heikeji/log', sess.graph)

    saver = tf.train.Saver(tf.all_variables())
    #last_epoch = load_model(sess, saver,'model/') 

    for epoch in range(10):
        sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))   

        for batche in range(n_chunk): 
            x,y = trainds.next_batch(batch_size)
            train_loss, _  = sess.run([merged_summaries, train_op], feed_dict={input_data: x, output_targets: y})  

            #all_loss = all_loss + train_loss 
            train_writer.add_summary(train_loss, batche+500*epoch)
            
            if batche % 50 == 1:
                print(epoch, 500*epoch+batche, 0.002 * (0.97 ** epoch)) 
            
        saver.save(sess, '/home/ccc/code/heikeji/model/poetry.module', global_step=epoch) 
        print (epoch)
    train_writer.close()
    sess.run()
        
train_model()  
