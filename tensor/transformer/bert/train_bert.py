#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: train_bert.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import numpy as np
import tensorflow as tf
import tensor.transformer.bert.bert as bert
import random


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_data_path', "/data/pre_train/train.tf",
                           'Training data path')

tf.app.flags.DEFINE_string('test_data_path', "/data/pre_train/test.tf",
                           'Test data path')

tf.app.flags.DEFINE_string('log_dir', "logs_bert", 'The log  dir')

tf.app.flags.DEFINE_string("wordvec_path", "matching/corpus/vec_160k_string.txt",
                           "the word word2vec data path")

tf.app.flags.DEFINE_integer("wordvec_size", 180, "the vec embedding size")

tf.app.flags.DEFINE_integer("max_tokens_per_sentence", 199,

                            "max num of tokens per sentence")

tf.app.flags.DEFINE_integer("max_epochs", 100, "max num of epoches")

tf.app.flags.DEFINE_integer("batch_size", 256, "num example per mini batch")

tf.app.flags.DEFINE_integer("step_size", 220000, "num example per mini batch")

tf.app.flags.DEFINE_integer("test_batch_size", 256,
                            "num example per test batch")

tf.app.flags.DEFINE_integer("train_steps", 4300000, "trainning steps")

tf.app.flags.DEFINE_integer("track_history", 15, "track max history accuracy")

tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

tf.app.flags.DEFINE_float("learning_rate_max", 0.004,
                          "the final minimal learning rate")

tf.app.flags.DEFINE_float("learning_rate_min", 0.0001,
                          "the final minimal learning rate")

# init word2vec and [SEP] [MASK]
def load_w2v(path, expectDim):
    fp = open(path, "r")
    print("load data from:%s" % (path))
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    print("total:%d, dim:%d" % (total, dim))
    assert (dim == expectDim)
    ws = []
    mv = [0 for i in range(dim)]
    for t in range(total):
        line = fp.readline()
        ss = line.split(" ")
        if len(ss) != (dim + 1):
            print("got error line:===========\n%s" % (line))
            assert(False)
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)
    for i in range(dim):
        mv[i] = mv[i] / total
    ws.append(mv)  # SEP
    rv = [random.random() for _ in range(dim)]
    ws.append(rv)  # MARK
    print("total vocab size:%d" % (len(ws)))

    fp.close()
    return np.asarray(ws, dtype=np.float32)


def make_feed_dict(model,
                   inputs,
                   droprate=0,
                   lr=None):
    dicts = {model.sentence_placeholder: inputs[0], model.label_target: inputs[1],
             model.label_index: inputs[2], model.a_length: inputs[3],
             model.ab_length: inputs[4], model.nsp_target: inputs[5],
             model.dropout_h: droprate}
    if lr is not None:
        dicts[model.learning_rate_h] = lr
    return dicts


def test_eval(sess, loss, acc_S, testDatas, batchSize, model):
    totalLen = len(testDatas)
    numBatch = int((totalLen - 1) / batchSize) + 1
    totalLoss = 0
    correct = 0
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        X = testDatas[i * batchSize:endOff]
        sentence = [x[0] for x in X]
        label_target = [x[1] for x in X]
        label_index = [x[2] for x in X]
        a_length = [x[3] for x in X]
        ab_length = [x[4] for x in X]
        nsp_target = [x[5] for x in X]
        inputs = [sentence, label_target, label_index,
                  a_length, ab_length, nsp_target]
        feed_dict = make_feed_dict(
            model, inputs, 0)
        lossv, correctV = sess.run([loss, acc_S], feed_dict)
        totalLoss += lossv
        correct += correctV
    return totalLoss / numBatch, float(correct) / totalLen


def load_test_dataset_all(sess, test_input, testDatas):
    loaded = 0
    while True:
        try:
            test = sess.run(test_input)
            testDatas.append(test)
            loaded += 1
        except tf.errors.OutOfRangeError:
            break


def main():
    curdir = os.path.dirname(os.path.realpath(__file__))
    trainDataPaths = tf.app.flags.FLAGS.train_data_path.split(",")
    for i in range(len(trainDataPaths)):
        if not trainDataPaths[i].startswith("/"):
            trainDataPaths[i] = curdir + "/../../" + trainDataPaths[i]

    testDataPaths = tf.app.flags.FLAGS.test_data_path.split(",")
    for i in range(len(testDataPaths)):
        if not testDataPaths[i].startswith("/"):
            testDataPaths[i] = curdir + "/../../" + testDataPaths[i]
    testDatas = []
    with tf.Graph().as_default():
        datasetTrain = tf.data.TFRecordDataset(
            trainDataPaths, compression_type='GZIP', buffer_size=4096*1024*10)
        datasetTrain = datasetTrain.map(
            bert.parse_tfrecord_function)
        datasetTrain = datasetTrain.repeat(FLAGS.max_epochs)
        datasetTrain = datasetTrain.shuffle(buffer_size=20000)
        datasetTrain = datasetTrain.batch(FLAGS.batch_size)
        iterator = datasetTrain.make_one_shot_iterator()
        batch_inputs = iterator.get_next()
        print("batch shape:%r" % (batch_inputs[2].get_shape()))

        datasetTest = tf.data.TFRecordDataset(
            testDataPaths, compression_type='GZIP', buffer_size=4096*1024*10)
        datasetTest = datasetTest.map(bert.parse_tfrecord_function)
        iteratorTest = datasetTest.make_initializable_iterator()

        test_input = iteratorTest.get_next()
        # 加载词向量
        wordsEm = load_w2v(FLAGS.wordvec_path, FLAGS.wordvec_size)
        model = bert.Model(FLAGS.max_tokens_per_sentence, wordsEm,
                           FLAGS.wordvec_size, len(wordsEm))
        print("train data path:", trainDataPaths)
        # loss
        loss, correct = model.loss()
        # train
        train_op = model.train(loss)
        # learning rate
        decayPerStep = (
            FLAGS.learning_rate - FLAGS.learning_rate_min) / FLAGS.train_steps

        sv = tf.train.Supervisor(logdir=FLAGS.log_dir)
        with sv.managed_session(master='') as sess:
            # actual training loop
            training_steps = FLAGS.train_steps
            bestLoss = float("inf")
            trackHist = 0
            sess.run(iteratorTest.initializer)
            # 加载测试数据集
            load_test_dataset_all(
                sess, test_input, testDatas)
            # write graph
            tf.train.write_graph(sess.graph_def,
                                 FLAGS.log_dir,
                                 "graph.pb",
                                 as_text=False)
            print("Loaded #tests:%d" % (len(testDatas)))
            # 开始训练
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    clipStep = int(step / 20000)
                    clipStep = clipStep * 20000
                    trainDatas = sess.run(batch_inputs)
                    lr = FLAGS.learning_rate - decayPerStep * clipStep
                    # build feed_dict
                    feedDict = make_feed_dict(
                        model, trainDatas, 0.2, lr)
                    trainLoss, _ = sess.run(
                        [loss, train_op], feedDict)
                    if (step + 1) % 100 == 0:
                        print("[%d] loss: [%r]" % (step + 1, trainLoss))
                    if (step + 1) % 2000 == 0 or step == 0:
                        tloss, acc = test_eval(
                            sess, loss, correct, testDatas,
                            FLAGS.test_batch_size, model)
                        print("test loss:%.3f,  acc:%.3f" %
                              (tloss, acc))
                        if step and tloss < bestLoss:
                            sv.saver.save(sess, FLAGS.log_dir + '/best_model')
                            trackHist = 0
                            bestLoss = tloss
                        else:
                            if trackHist >= FLAGS.track_history:
                                print(
                                    "always not good enough in last %d histories, best accuracy:%.3f"
                                    % (trackHist, bestLoss))
                                break
                            else:
                                trackHist += 1
                except KeyboardInterrupt, e:
                    sv.saver.save(sess,
                                  FLAGS.log_dir + '/model',
                                  global_step=(step + 1))
                    raise e
            sv.saver.save(sess, FLAGS.log_dir + '/finnal-model')


if __name__ == '__main__':
    tf.app.run()
