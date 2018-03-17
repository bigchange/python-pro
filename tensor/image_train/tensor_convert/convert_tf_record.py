import fire,os
import tensorflow as tf
from PIL import Image
import numpy as np

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

def convert2gray(img):
    # print ("image shape:", image.shape)
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

MAX_CAPTCHA = 4
# 63
char_set = number + alphabet + ALPHABET + ['_']

CHAR_SET_LEN = len(char_set)

def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('length max 4')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

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

def load_data(trainPath, writer):
    files = os.listdir(trainPath)
    for fi in files:
        fi_d = os.path.join(trainPath, fi)
        if os.path.isdir(fi_d):
            continue
        else:
            file_path = os.path.join(trainPath, fi_d)
            imageInput = Image.open(file_path)
            basePath = os.path.basename(file_path)
            text = os.path.splitext(os.path.basename(basePath))[0]
            imageInput = imageInput.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            imageInput = np.array(imageInput)
            imageInput = convert2gray(imageInput)
            image = imageInput.flatten() / 255
            image = image.tolist()
            target = text2vec(text)
            target = target.tolist()
            example = tf.train.Example(features=tf.train.Features(feature={
                "target": tf.train.Feature(float_list=tf.train.FloatList(value=target)),
                "image": tf.train.Feature(float_list=tf.train.FloatList(value=image)),
            }))
            writer.write(example.SerializeToString())

def convert(trainPath,
            trainOutPath,
            testPath,
            testOutPath):
    writerTrain = tf.python_io.TFRecordWriter(trainOutPath)
    writerTest = tf.python_io.TFRecordWriter(testOutPath)

    load_data(trainPath, writerTrain)
    load_data(testPath, writerTest)

    writerTrain.close()
    writerTest.close()


if __name__ == '__main__':
    fire.Fire()

"""

python convert_tf_record.py convert --trainPath=/Users/devops/Downloads/github/codeRec/gen_captcha/train --trainOutPath=/Users/devops/Downloads/github/codeRec/gen_captcha/train_tf/train.tfrecord --testPath=/Users/devops/Downloads/github/codeRec/gen_captcha/test --testOutPath=/Users/devops/Downloads/github/codeRec/gen_captcha/test_tf/test.tfrecord

"""