# coding:utf-8
import os
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random, time

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

data_path = "/Users/devops/Downloads/github/codeRec/gen_captcha/"

train_path = "train/"

test_path = "test/"

list_path = "data/"


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

    print("return:", filepaths, labels)
    return filepaths, labels

# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # 写到文件
    number = random.randint(0, 100)
    if number < 10:
        image.write(captcha_text, data_path + train_path + captcha_text + '.jpg')
    else:
        image.write(captcha_text, data_path + test_path + captcha_text + '.jpg')
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


if __name__ == '__main__':
    print 'begin ', time.ctime()
    pic_size = 0
    # generate code
    # while(1):
    #     text, image = gen_captcha_text_and_image()
    #     if pic_size > 100000:
    #         break
    #     pic_size = pic_size + 1
    read_label_file(data_path + list_path)
    print 'end ', time.ctime()
