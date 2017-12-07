# -*- coding: utf-8 -*-
from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET
import fire

import numpy as np
import tensorflow as tf

text, image = gen_captcha_text_and_image()
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(text)


def parse_tfrecord_function(example_proto):
    features = {
        "image": tf.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True),
        "target": tf.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True),

    }
    parsed_features = tf.parse_single_example(example_proto, features)
    target = parsed_features["target"]
    target.set_shape([252])
    image = parsed_features["image"]
    image.set_shape([9600])
    # image = tf.reshape(image, [IMAGE_HEIGHT * IMAGE_WIDTH])
    # print ("image tensor shape:",image.shape)
    # image.set_shape([None, IMAGE_HEIGHT * IMAGE_WIDTH])
    return target, image


def convert2gray(img):
    # print ("image shape:", image.shape)
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


# 63
char_set = number + alphabet + ALPHABET + ['_']

CHAR_SET_LEN = len(char_set)


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('length max 4')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    # get char pos in vocabulary
    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def transformX(image):
    image = convert2gray(image)

    return image.flatten() / 255  # (image.flatten()-128)/128  mean为0


def transformY(text):
    return text2vec(text)


def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c / 63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


####################################################################
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


def load_test_dataset_all(sess, test_input, testDatas):
    while True:
        try:
            target, image = sess.run(test_input)
            testDatas.append(
                [target, image])
        except tf.errors.OutOfRangeError:
            break


def test_eval(sess, accuracy, testDatas, batchSize):
    totalLen = len(testDatas)
    numBatch = int((totalLen - 1) / batchSize) + 1
    totalAccuracy = 0
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        X_val = testDatas[i * batchSize:endOff]
        feed_dict = {X: [item[1] for item in X_val], keep_prob: 1.}
        accuracyV = sess.run([accuracy], feed_dict)
        totalAccuracy += accuracyV
    return totalAccuracy / numBatch


def load_batch_data(train_path, test_path):
    datasetTrain = tf.contrib.data.TFRecordDataset(train_path)
    datasetTrain = datasetTrain.map(parse_tfrecord_function)
    datasetTrain = datasetTrain.repeat(10)
    datasetTrain = datasetTrain.shuffle(buffer_size=200)
    datasetTrain = datasetTrain.batch(64)
    iterator = datasetTrain.make_one_shot_iterator()
    batch_inputs = iterator.get_next()

    datasetTest = tf.contrib.data.TFRecordDataset(test_path)
    datasetTest = datasetTest.map(parse_tfrecord_function)
    datasetTest = datasetTest.repeat(10)
    datasetTest = datasetTest.shuffle(buffer_size=200)
    iteratorTest = datasetTest.make_initializable_iterator()
    test_inputs = iteratorTest.get_next()
    return batch_inputs, test_inputs, iteratorTest


# 训练
def train_crack_captcha_cnn(trainDataPaths, testDataPath):
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print("step:%s,loss:%s" % (step, loss_))

            if step % 100 == 0:
                test_image, test_label = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: test_image, Y: test_label, keep_prob: 1.})
                print(step, acc)
                if acc >= 0.9:
                    saver.save(sess, "crack_capcha.model", global_step=step)
                    break
                if acc > 0.98:
                    saver.save(sess, "crack_capcha.model", global_step=step)
                    break
            step += 1


def main():
    train_crack_captcha_cnn("train.tfrecord", "test.record")
    # fire.Fire()


if __name__ == '__main__':
    main()
