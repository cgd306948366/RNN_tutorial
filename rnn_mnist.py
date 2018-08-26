import os
import tensorflow as tf
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist/', one_hot=True)
batch_size = 64
width = 28
height = 28
rnn_size = 256
out_size = 10

def weight_variable(shape, w_alpha=0.01):
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variable(shape, b_alpha=0.1):
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)

def rnn_graph(x, rnn_size, out_size, width, height):
    # weight and bias setting
    w = weight_variable([rnn_size, out_size])
    b = bias_variable([out_size])
    # LSTM
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, width])
    # split the dataset to height step, one step is batch*width
    x = tf.split(x, height)
    # We only need the last output，this model is a many to one model
    outputs, status = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
    y_conv = tf.add(tf.matmul(outputs[-1], w), b)
    return y_conv

def optimize_graph(y, y_conv):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    return optimizer

def accuracy_graph(y, y_conv):
    correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

def train():
    x = tf.placeholder(tf.float32, [None, height, width])
    y = tf.placeholder(tf.float32)
    y_conv = rnn_graph(x, rnn_size, out_size, width, height)
    optimizer = optimize_graph(y, y_conv)
    accuracy = accuracy_graph(y, y_conv)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        acc_rate = 0.99
        while True:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape([batch_size, height, width])
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % 50 == 0:
                batch_x_test = mnist.test.images
                batch_y_test = mnist.test.labels
                batch_x_test = batch_x_test.reshape([-1, height, width])
                acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test})
                print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:', acc)
                if acc >= acc_rate:
                    model_path = os.getcwd() + os.sep + "mnist.model"
                    saver.save(sess, model_path, global_step=step)
                    break
            step += 1

if __name__ == '__main__':
    train()