
import tensorflow as tf
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET

char_set = number + alphabet + ALPHABET + ['_']

CHAR_SET_LEN = len(char_set)

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4

def parse_tfrecord_function(example_proto):
    features = {
        "image": tf.FixedLenFeature([], tf.float32, default_value=0),
        "target": tf.FixedLenFeature([], tf.float32, default_value=0),

    }
    parsed_features = tf.parse_single_example(example_proto, features)
    target = parsed_features["target"]
    target.set_shape([MAX_CAPTCHA * CHAR_SET_LEN])
    image = parsed_features["image"]
    image.set_shape([IMAGE_HEIGHT * IMAGE_WIDTH])
    return target, image

def get_input(path):
    datasetTrain = tf.contrib.data.TFRecordDataset(path)
    datasetTrain = datasetTrain.map(parse_tfrecord_function)
    datasetTrain = datasetTrain.repeat(10)
    datasetTrain = datasetTrain.shuffle(buffer_size=200)
    datasetTrain = datasetTrain.batch(64)
    iterator = datasetTrain.make_one_shot_iterator()
    batch_inputs = iterator.get_next()
    return batch_inputs