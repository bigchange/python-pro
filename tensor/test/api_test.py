import tensorflow as tf
import numpy as np


x = tf.constant([[1., 1.], [2., 2.]])


def length(data, axis=1):
    used = tf.sign(tf.abs(data))
    length = tf.reduce_sum(used, reduction_indices=axis)
    length = tf.cast(length, tf.int32)
    return length

with tf.Session() as sess:
    data = [[1,3,5], [2,4,0], [1,5,7]]
    lengthMask = tf.cast(
        tf.sequence_mask(length(data),4), tf.float32)
    print ("length:", sess.run(lengthMask))

