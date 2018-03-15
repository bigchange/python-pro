# -*- coding:utf-8 -*-
import tensorflow as tf
import time

sess = tf.Session()
A = tf.get_variable(
    name="attention",
    shape=[2, 2, 4],
    initializer=tf.random_uniform_initializer(-1, 1))
init = tf.global_variables_initializer()
sess.run(init)


# Test: softmax #
def test_softmax(A):
    print ("Test: softmax")
    print("A:", sess.run(A))
    print ("ExpA:", sess.run(tf.exp(A)))
    print ("ReduceSum:", sess.run(tf.reduce_sum(tf.exp(A), axis=-1)))
    A = tf.nn.softmax(A)
    print ("retA", sess.run(A))
    retA = tf.nn.softmax_cross_entropy_with_logits(labels=A, logits=A)
    print ("coss entropy:", sess.run(retA))

# [ actual lengths for each of the sequences in the batch ]
def length():
    data = tf.constant([[1, 2, 3, 4, 5, 6], [1, 2, 4, 5, 6, 7]])
    used = tf.sign(tf.abs(data))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

L = length()


def string2Char(str):
    chars = list(unicode(str.decode("utf8")))
    # [我爱你] -> [u'\u6211', u'\u7231', u'\u4f60']
    # [ I LOVE YOU] -> [u'I', u' ', u'L', u'O', u'V', u'E', u' ', u'Y', u'O', u'U']
    return chars

print (string2Char("I LOVE YOU"))
