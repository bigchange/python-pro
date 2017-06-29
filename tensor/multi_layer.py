import tensorflow as tf
import numpy as np

image_size = 28
batch_size = 128
fc1_size = 4096
fc2_size = 2048
fc3_size = 128
num_labels = 10

valid_dataset = []
valid_labels = []
test_dataset = []
test_labels = []
train_dataset = []
train_labels = []


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    tf_beta = tf.placeholder(tf.float32)
    global_step = tf.Variable(0)  # count the number of steps taken.

    # Variables.
    # stddev is very important!!!
    W1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, fc1_size], stddev=np.sqrt(2.0 / (image_size * image_size))))
    b1 = tf.Variable(tf.zeros([fc1_size]))

    W2 = tf.Variable(tf.truncated_normal([fc1_size, fc2_size], stddev=np.sqrt(2.0 / fc1_size)))
    b2 = tf.Variable(tf.zeros([fc2_size]))

    W3 = tf.Variable(tf.truncated_normal([fc2_size, fc3_size], stddev=np.sqrt(2.0 / fc2_size)))
    b3 = tf.Variable(tf.zeros([fc3_size]))

    W4 = tf.Variable(tf.truncated_normal([fc3_size, num_labels], stddev=np.sqrt(2.0 / fc3_size)))
    b4 = tf.Variable(tf.zeros([num_labels]))
    # Training computation.
    y1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
    # y1 = tf.nn.dropout(y1, 0.5)

    y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
    # y2 = tf.nn.dropout(y2, 0.5)

    y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
    # y3 = tf.nn.dropout(y3, 0.5)

    logits = tf.matmul(y3, W4) + b4

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # data loss + weight loss
    loss = loss + tf_beta * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) +
                             tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(b4))

    # Optimizer
    learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.7, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    y1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
    y2_valid = tf.nn.relu(tf.matmul(y1_valid, W2) + b2)
    y3_valid = tf.nn.relu(tf.matmul(y2_valid, W3) + b3)
    valid_logits = tf.matmul(y3_valid, W4) + b4
    valid_prediction = tf.nn.softmax(valid_logits)

    y1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
    y2_test = tf.nn.relu(tf.matmul(y1_test, W2) + b2)
    y3_test = tf.nn.relu(tf.matmul(y2_test, W3) + b3)
    test_logits = tf.matmul(y3_test, W4) + b4
    test_prediction = tf.nn.softmax(test_logits)

# Let's run it:
num_steps = 12001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_beta: 0.001438}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Final Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))