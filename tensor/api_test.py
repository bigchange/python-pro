import tensorflow as tf
import numpy as np

w = tf.constant([[1,-0.5],[0.6,-0.4]], dtype=tf.float32)
with tf.Session() as sess:
    print ("w:", sess.run(w))
    print ("r:", sess.run(tf.nn.relu(w)))
    print ("out:", sess.run(tf.nn.dropout(w, keep_prob=0.5)))
