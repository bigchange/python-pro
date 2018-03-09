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

print ("L:", sess.run(L))
