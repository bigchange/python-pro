#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: convert_to_tfrecord.py
# Project: bert

import os
import tensorflow as tf
import fire
import sentencepiece as spm
import random
import json
import numpy as np

MAX_TOKEN_NUM_PER_SENTENCE = 199
MAX_LABELS = 26

# SEP --> totalVocab
# MASK --> totalVocab+1
def gen_sentence_features(a, b, seg):
    a = a.lower()
    b = b.lower()
    totalVocab = seg.GetPieceSize()
    ss = seg.EncodeAsIds(a)
    na = len(ss)
    ss.append(totalVocab)  # SEP
    ss += seg.EncodeAsIds(b)
    nab = len(ss)
    ss.append(totalVocab)  # SEP
    if nab > (MAX_TOKEN_NUM_PER_SENTENCE - 1) or na < 3 or nab <6:
        return None, None, None, None, None
    label_index = []
    label_target = []
    MASK = totalVocab+1
    # label index should plus one, will pad '[CLS]' during trainning
    for i in range(na):
        # 15% 的用来mask
        if random.random() <= 0.15 and len(label_index)<MAX_LABELS:
            # label_index 已经是有序的
            label_index.append(i+1)
            # 对应label_index下的target的encode_id(spm)
            label_target.append(ss[i])
            # 80% mask
            if random.random() <= 0.8:
                ss[i] = MASK
            else:
                # 50% 随机
                if random.random() <= 0.5:
                    # random one
                    ss[i] = random.randint(1, totalVocab-1)
    for i in range(na+1, nab):
        if random.random() <= 0.15 and len(label_index)<MAX_LABELS:
            label_index.append(i+1)
            label_target.append(ss[i])
            if random.random() <= 0.8:
                ss[i] = MASK
            else:
                if random.random() <= 0.5:
                    # random one
                    ss[i] = random.randint(1, totalVocab-1)
    # 不够最大长度，补0
    for i in range(nab+1, MAX_TOKEN_NUM_PER_SENTENCE):
        ss.append(0)

    assert(len(label_index)==len(label_target))

    # 不够最大长度，补0
    for i in range(len(label_index), MAX_LABELS):
      label_index.append(0)
      label_target.append(0)

    assert(len(ss)==MAX_TOKEN_NUM_PER_SENTENCE)

    assert(len(label_index)==MAX_LABELS)

    assert(len(label_target)==MAX_LABELS)

    return ss, na, nab, label_index, label_target


def convert(trainPath,
            trainOutPath,
            testOutPath,
            spModelPath,
            testRatio=0.004):
    options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    writerTrain = tf.python_io.TFRecordWriter(trainOutPath, options=options)
    writerTest = tf.python_io.TFRecordWriter(testOutPath, options=options)
    seg = spm.SentencePieceProcessor()
    seg.Load(spModelPath)
    npos = 0
    nneg = 0
    skiped = 0
    processed = 0
    with open(trainPath, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            processed += 1
            ss = line.split("\x01", 3)
            assert (len(ss) == 3)
            nsp_target = int(ss[2])
            ss, na, nab, label_index, label_target = gen_sentence_features(
                ss[0], ss[1], seg)
            if ss is not None:
                example = tf.train.Example(features=tf.train.Features(feature={
                    "sentence": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=ss)),
                    'label_target': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=label_target)),
                    'label_index': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=label_index)),
                    'a_length': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[na])),
                    'ab_length': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[nab])),
                    'nsp_target': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[nsp_target])),
                }))
                if random.random() <= testRatio:
                    writerTest.write(example.SerializeToString())
                    nneg += 1
                else:
                    writerTrain.write(example.SerializeToString())
                    npos += 1
            else:
                skiped += 1
            if processed % 10000 == 0:
                print("processed %d, neg:%d, pos:%d, skip:%d....." %
                      (processed, nneg, npos, skiped))
    print("processed %d, neg:%d, pos:%d, skip:%d....." %
          (processed, nneg, npos, skiped))
    writerTrain.close()
    writerTest.close()


if __name__ == '__main__':
    fire.Fire()
