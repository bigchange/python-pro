import tensorflow as tf
import numpy as np


x = tf.constant([[1., 1.], [2., 2.]])

with tf.Session() as sess:
    print ("result:", sess.run([tf.reduce_mean(x), tf.reduce_mean(x, 0, keepdims=True),tf.reduce_mean(x, 1, keepdims=True)]))
