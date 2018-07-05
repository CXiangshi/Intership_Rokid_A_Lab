import tensorflow as tf
import numpy as np
import os
import collections
import dataset



def input():

    with tf.name_scope('input'):
        input_data = tf.placeholder(tf.int32, [64, None])
        output_targets = tf.placeholder(tf.int32, [64, None])

    return input_data, output_targets


def rnn_model(rnn_size, nums_layers, input_data, output_targets): 

    with tf.name_scope('rnn_lstm'):
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell
        cell = cell_fun(rnn_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*nums_layers, state_is_tuple=True)

        if output_targets is not None:
            initial_state = cell.zero_state(dataset.batch_size, tf.float32)
        else:
            initial_state = cell.zero_state(1, tf.float32)
    
    with tf.name_scope('lm'):
        embedding = tf.get_variable('embedding', [len(dataset.words), rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, input_data)

        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        output = tf.reshape(outputs, [-1, rnn_size])
    
    with tf.name_scope('softmax'):
        softmax_weights = tf.get_variable('softmax_w', [rnn_size, len(dataset.words)])
        softmax_bias = tf.get_variable('softmax_b', [len(dataset.words)])
        logits = tf.nn.bias_add(tf.matmul(output, softmax_weights), softmax_bias)
    
    with tf.name_scope('learning_rate'):
        learning_rate = tf.Variable(0.001, trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)

    end_points = dict()

    if output_targets is not None:
        
        labels = tf.one_hot(tf.reshape(output_targets, [-1]), depth=len(dataset.words))
        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            total_loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', total_loss)

        with tf.name_scope('train'):
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.apply_gradients(zip(grads, tvars))
        
        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['total_loss'] = total_loss
        end_points['last_state'] = last_state
        end_points['train_op'] = train_op
    else:
        probs = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = probs

    return end_points

def train_model():
    tf.logging.set_verbosity(tf.logging.INFO)
    
    trainds = dataset.DataSet(len(dataset.poetrys_vector))
    sess = tf.InteractiveSession()

    input_data , targets = input()

    end_points = rnn_model(128, 2, input_data, targets)

    tf.global_variables_initializer().run()
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.getcwd() + '/log', sess.graph)

    saver = tf.train.Saver(tf.global_variables())

    for epoch in range(10): 

        for steps in range(dataset.n_chunk): 
            x,y = trainds.next_batch(dataset.batch_size)
            summary, loss,  _  = sess.run([merged_summaries, end_points['train_op'], end_points['total_loss']], feed_dict={input_data: x, targets: y})  
            print(loss)
            train_writer.add_summary(summary, 500*epoch+steps)

            tf.logging.info('steps:%d,loss:%.1f'%(500*epoch+steps, loss))
            
            #if batche % 50 == 1:
                #print(epoch, 500*epoch+batche, 0.002 * (0.97 ** epoch)) 
            
        saver.save(sess, os.getcwd() + '/model', global_step=epoch) 
        print (epoch)
    train_writer.close()
    sess.close()
#train_model()