import tensorflow as tf
import numpy as np
from rnn_mnist import rnn_graph
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist/', one_hot=True)
width = 28
height = 28
rnn_size = 256
out_size = 10

txtName = "rnn_test_result.txt"

def mnistTotext(image_list, height, width, rnn_size, out_size):
    x = tf.placeholder(tf.float32, [None, height, width])
    y_conv = rnn_graph(x, rnn_size, out_size, width, height)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(y_conv, 1)
        vector_list = sess.run(predict, feed_dict={x: image_list})
        vector_list = vector_list.tolist()
        return vector_list

if __name__ == '__main__':
    batch_x_test = mnist.test.images
    batch_x_test = batch_x_test.reshape([-1, height, width])
    batch_y_test = mnist.test.labels
    batch_y_test = list(np.argmax(batch_y_test, 1))
    pre_y = list(mnistTotext(batch_x_test, height, width, rnn_size, out_size))
    with open(txtName,'w+') as f:
        for text in batch_y_test:
            result = 'Label:' + str(text) + ' Predict:'+ str(pre_y[batch_y_test.index(text)])
            print(result)
            f.write(result)
            f.write('\n')