
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[ ]:


class TrainModel(object):
    
    def __init__(self, data, labels, emb_keep, rnn_keep):
        self.data = data
        self.labels = labels
        self.emb_keep = emb_keep
        self.rnn_keep = rnn_keep
        self.global_step
        self.cell
        self.predict
        self.loss
        self.optimize
    
    def cell(self):
        lstm_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(128), output_keep_prob=self.rnn_keep) for _ in range(2)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        return cell
    
    def predict(self):
        embedding = tf.get_variable('embedding', shape=[6272, 128])
        if True:
            softmax_weights = tf.transpose(embedding)
        else:
            softmax_weights = tf.get_variable('softmaxweights', shape=[128, 6272])
        softmax_bias = tf.get_variable('softmax_bias', shape=[6272])
        emb = tf.nn.embedding_lookup(embedding, self.data)
        emb_dropout = tf.nn.dropout(emb, self.emb_keep)
        self.init_state = self.cell.zero_state(64, dtype=tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(self.cell, emb_dropout, emb_dropout, scope='d_rnn', dtype=tf.float32, initial_state=self.init_state)
        outputs = tf.reshape(outputs, [-1, 128])
        logits = tf.matmul(outputs, softmax_weights) + softmax_bias
        return logits
    
    def loss(self):
        outputs_target = tf.reshape(self.labels, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict, labels=outputs_target)
        cost = tf.reduce_mean(loss)
        return cost
    
    def global_step(self):
        global_step = tf.Variable(0, trainable=False)
        return global_step
    
    def optimize(self):
        learn_rate = tf.train.exponential_decay(0.0005, self.global_step, 600, 0.92)
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables),5.0)
        optimizer = tf.train.AdamOptimizer(learn_rate)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables), self.global_step)
        return train_op
    
class EvalModel(object):
    
    def __init__(self, data, emb_keep, rnn_keep):
        self.data = data
        self.emb_keep = emb_keep
        self.rnn_keep = rnn_keep
        self.cell
        self.predict
        self.prob

    def cell(self):
        lstm_cell = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=self.rnn_keep) for
            _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        return cell
        
    def predict(self):
        embedding = tf.get_variable('embedding', shape=[6272, 128])
        if True:
            softmax_weights = tf.transpose(embedding)
        else:
            softmax_weights = tf.get_variable('softmaxweights', shape=[128, 6272])
        softmax_bias = tf.get_variable('softmax_bias', shape=[6272])
        
        emb = tf.nn.embedding_lookup(embedding, self.data)
        emb_dropout = tf.nn.dropout(emb, self.emb_keep)
        self.init_state = self.cell.zero_state(1, dtype=tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(self.cell, emb_dropout, scope='d_rnn', dtype=tf.float32, initial_state=self.init_state)
        outputs = tf.reshape(outputs, [-1, 128])
        
        logits = tf.matmul(outputs, softmax_weights) + softmax_bias
        self.last_state = last_state
        return logits
    
    def prob(self):
        probs = tf.nn.softmax(self.predict)
        return probs
        

