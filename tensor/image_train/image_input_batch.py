# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.
import tensorflow as tf
import random
import os
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET

import numpy as np
import tensorflow as tf

text, image = gen_captcha_text_and_image()
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(text)

def transformX(image):
    image = convert2gray(image)
    return image / 255

def transformY(text):
    return text2vec(text)

def convert2gray(img):
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

dataset_path = "/Users/devops/Downloads/github/codeRec/gen_captcha/"
test_labels_file = "test/"
train_labels_file = "train/"

test_set_size = 5

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
NUM_CHANNELS = 3
BATCH_SIZE = 64

def decode_lable(file):
    return os.path.splitext(os.path.basename(file))[0]

def encode_label(label):
    return int(label)


def read_label_file(path):
    filepaths = []
    labels = []
    files = os.listdir(path)
    for fi in files:
        fi_d = os.path.join(path, fi)
        if os.path.isdir(fi_d):
            continue
        else:
            file_path = os.path.join(path, fi_d)
            filepaths.append(file_path)
            labels.append(decode_lable(fi_d))
    return filepaths, labels


def get_batch_size():

    return image_trains_batch,train_label_batch,test_image_batch, test_label_batch

if __name__ == '__main__':
    imageDefine, textDefine, _ ,_ = get_batch_size()
    print ("shape:", textDefine.shape)
    print ("image:", image)
    gray = np.mean(image, -1)