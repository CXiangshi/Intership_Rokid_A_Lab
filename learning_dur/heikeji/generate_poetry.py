import tensorflow as tf 
import dataset
import numpy as np
import os
from model import rnn_model, input



def generate_p(heads=None):
    
    def to_word(predict, words):
        sample = np.argmax(predict)
        if sample > len(dataset.words):
            sample = len(dataset.words) - 1
        return dataset.words[sample]

    tf.reset_default_graph()
    input_data1 = tf.placeholder(tf.int32, [1, None])
    end_points = rnn_model(rnn_size=128, nums_layers=2, input_data=input_data1, output_targets=None)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        tf.global_variables_initializer().run()

        ckpt = tf.train.latest_checkpoint(os.getcwd() + '/model/')
        saver.restore(sess, ckpt)
        peom = ''
        x = np.array([list(map(dataset.word_num_map.get, u'['))])
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],feed_dict={input_data1: x})
        word = to_word(predict, dataset.words)

        for head in heads:
            #x = np.array([list(map(dataset.word_num_map.get, u'['))])
            #[predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],feed_dict={input_data1: x})
            
            sentence = ''
            sentence = head
            x[0, 0] = dataset.word_num_map[sentence]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],feed_dict={input_data1: x, end_points['initial_state']:last_state})

            #x = np.zeros((1, 1))
           
            #word = to_word(predict, dataset.words)
            #sentence += word
            while word !=u'。':
                
                x = np.zeros((1, 1))
                x[0, 0] = dataset.word_num_map[word]
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],feed_dict={input_data1: x, end_points['initial_state']: last_state})
                word = to_word(predict, dataset.words)
                sentence += word
            peom += sentence
            peom += u'\n'    
    return peom

print(generate_p(u''))