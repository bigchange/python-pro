import tensorflow as tf
from tensorflow.tensorboard.tensorboard import FLAGS
import numpy as np

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# load data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Weight Initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# First Convolutional Layer

W_conv1 = weight_variable([5, 5, 1, 32])  # 5 * 5 * 32
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1], name="x-image")  # 28 * 28 * 1

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="layer-one-relu")  # 28 * 28 * 32
h_pool1 = max_pool_2x2(h_conv1)  # 14 * 14 * 64

# Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])  # 28 * 28 * 64
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name="layer-two-relu")  # 14 * 14 * 64
h_pool2 = max_pool_2x2(h_conv2)  # 7 * 7 * 64

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1, name="layer-three-matmul") + b_fc1,
                   name="layer-three-relu")

# Dropout
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="drop-out")

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2, name="layer-four") + b_fc2

# Train and Evaluate the Mode
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv,
                                                        name="cross-entropy")
beta_regul = tf.placeholder(tf.float32)
beta_val = np.logspace(-4, -2, 20)

loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

# + 正则化 = + beta_regul * tf.nn.l2_loss(weights)

tf.summary.scalar("loss", loss)

FLAGS.train_dir = "/Users/devops/workspace/tensorflow/train"

ckpt_path = "/Users/devops/workspace/tensorflow/ckpt_dir/ckpt"

with tf.Session() as sess:
    # checkout point
    saver = tf.train.Saver(name="saver")
    # TensorBoard writer
    # visualize tensor board using command: tensorboard --logdir=path/to/log-directory
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    summary = tf.summary.merge_all()

    train_step = tf.train.AdamOptimizer(1e-4, name="adam").minimize(cross_entropy, name="minimize")
    correct_prediction = tf.equal(tf.argmax(y_conv, 1, name="y_conv"), tf.argmax(y_, 1, name="y_"),
                                  name="correct_prediction")
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32, name="cast_correct_prediction"),
        name="accuracy")

    # init variables
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = mnist.train.next_batch(50)
        # TensorBoard
        summary_str, _ = sess.run([summary, train_step],
                                  feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5,
                                             beta_regul:beta_regul})
        summary_writer.add_summary(summary_str, i)
        # saving ckpt
        saver.save(sess, ckpt_path, global_step=i)
        # saver.restore(sess, FLAGS.train_dir)  # restore model from check point file

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

            # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
