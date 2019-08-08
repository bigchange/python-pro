#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: train_use_official.py
# Copyright 2020 - 2019


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import tensorflow as tf

from tensor import Model as bert_model

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_data_path', "/data/pre_train/train.tf",
                           'Training data path')
tf.app.flags.DEFINE_string('test_data_path', "/data/pre_train/test.tf",
                           'Test data path')
tf.app.flags.DEFINE_string('log_dir', "logs_cbert", 'The log  dir')
tf.app.flags.DEFINE_string('best_dir', "best_cbert", 'The best log  dir')
tf.app.flags.DEFINE_integer("wordvec_size", 180, "the vec embedding size")
tf.app.flags.DEFINE_integer("vocab_size", 32003, "the vocab size")
tf.app.flags.DEFINE_integer("max_tokens_per_sentence", 199,
                            "max num of tokens per sentence")

tf.app.flags.DEFINE_integer("max_epochs", 100, "max num of epoches")
tf.app.flags.DEFINE_integer("batch_size", 512, "num example per mini batch")
tf.app.flags.DEFINE_integer("step_size", 100000, "num example per mini batch")
tf.app.flags.DEFINE_integer("warm_steps", 10000, "warm steps for training")
tf.app.flags.DEFINE_integer("test_batch_size", 512,
                            "num example per test batch")
tf.app.flags.DEFINE_integer("train_steps", 320000, "trainning steps")
tf.app.flags.DEFINE_integer("track_history", 15, "track max history accuracy")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_float("droprate", 0.1, "droprate")
tf.app.flags.DEFINE_float("learning_rate_max", 0.004,
                          "the final minimal learning rate")
tf.app.flags.DEFINE_float("learning_rate_min", 0.0001,
                          "the final minimal learning rate")


def make_feed_dict(model,
                   inputs,
                   droprate=0):
    dicts = {
         model.sentence_placeholder: inputs[0],
         model.label_target: inputs[1],
         model.label_index: inputs[2],
         model.totalLength: inputs[3],
         model.a_length: inputs[4],
         model.target: inputs[5],
         model.dropout_h: droprate,
    }
    return dicts


def test_eval(sess, loss, sameCountS, acc_S, acc_S2, testDatas, batchSize, model):
    totalLen = len(testDatas)
    numBatch = int((totalLen - 1) / batchSize) + 1
    totalLoss = 0
    totalAcc = 0
    totalSame = 0
    totalLabels = 0
    totalAcc2 = 0
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        X = testDatas[i * batchSize:endOff]
        sentence = [x[0] for x in X]
        label_target = [x[1] for x in X]
        label_index = [x[2] for x in X]
        total_length = [x[3] for x in X]
        a_length = [x[4] for x in X]
        target = [x[5] for x in X]

        for x in X:
            gotF = False
            for k in range(len(x[2])):
                if x[2][k] == 0:
                    totalLabels += k
                    gotF = True
                    break
            if not gotF:
                totalLabels += len(x[2])
        inputs = [sentence, label_target, label_index,
                  total_length, a_length, target]
        feed_dict = make_feed_dict(
            model, inputs, 0)
        lossv, samev, accv, accv2 = sess.run(
            [loss, sameCountS, acc_S, acc_S2], feed_dict)
        totalAcc += accv
        totalSame += samev
        totalLoss += lossv
        totalAcc2 += accv2
    return totalLoss / numBatch, totalAcc/(numBatch+0.000001), totalSame / (totalLabels+0.0000001), totalAcc2/(numBatch+0.000001),


def load_test_dataset_all(sess, test_input, testDatas):
    loaded = 0
    while True:
        try:
            test = sess.run(test_input)
            if random.random() < 0.4:
                testDatas.append(test)
                loaded += 1
        except tf.errors.OutOfRangeError:
            break


def main(unused_argv):
    curdir = os.path.dirname(os.path.realpath(__file__))
    # 可以逗号分隔 - 添加多个训练数据的路径
    trainDataPaths = tf.app.flags.FLAGS.train_data_path.split(",")
    for i in range(len(trainDataPaths)):
        if not trainDataPaths[i].startswith("/"):
            # 当前文件所在目录的上面两级目录下的路径
            trainDataPaths[i] = curdir + "/../../" + trainDataPaths[i]

    testDataPaths = tf.app.flags.FLAGS.test_data_path.split(",")
    for i in range(len(testDataPaths)):
        # 同理train
        if not testDataPaths[i].startswith("/"):
            testDataPaths[i] = curdir + "/../../" + testDataPaths[i]
    testDatas = []
    with tf.Graph().as_default():
        # tfrecord形式的数据读取过程的一种方法
        filenames = tf.data.Dataset.list_files(trainDataPaths)
        datasetTrain = filenames.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(
                    filename, compression_type='GZIP', buffer_size=4096*1024),
                cycle_length=10))
        # datasetTrain = tf.data.TFRecordDataset(
        #     trainDataPaths, compression_type='GZIP', buffer_size=4096*1024*10)
        # tfrecord数据的解码函数: parse_tfrecord_function
        datasetTrain = datasetTrain.map(bert_model.parse_tfrecord_function)
        datasetTrain = datasetTrain.repeat(FLAGS.max_epochs)
        datasetTrain = datasetTrain.shuffle(buffer_size=10000)
        datasetTrain = datasetTrain.batch(FLAGS.batch_size)
        iterator = datasetTrain.make_one_shot_iterator()
        batch_inputs = iterator.get_next()
        print("batch shape:%r" % (batch_inputs[2].get_shape()))

        # tfrecord形式的数据读取过程的另一种方法
        datasetTest = tf.data.TFRecordDataset(
            testDataPaths, compression_type='GZIP', buffer_size=4096*1024*10)
        datasetTest = datasetTest.map(
            bert_model.parse_tfrecord_function)
        iteratorTest = datasetTest.make_initializable_iterator()
        test_input = iteratorTest.get_next()
        model = bert_model.Model(FLAGS.wordvec_size, FLAGS.vocab_size)
        print("train data path:", trainDataPaths)
        # jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
        # with jit_scope():
        loss, sameCountS, acc_S, acc_S2 = model.loss(training=True)
        loss_t, sameCountS_t, acc_S_t, acc_S2_t = model.loss(training=False)
        train_op = model.train(loss, FLAGS.learning_rate,
                               FLAGS.train_steps, FLAGS.warm_steps)
        decayPerStep = (
            FLAGS.learning_rate - FLAGS.learning_rate_min) / FLAGS.train_steps
        bestSaver = tf.train.Saver()
        sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_model_secs=3800)
        with sv.managed_session(master='') as sess:
            # actual training loop
            training_steps = FLAGS.train_steps
            bestLoss = float("inf")
            trackHist = 0
            sess.run(iteratorTest.initializer)
            load_test_dataset_all(
                sess, test_input, testDatas)
            tf.train.write_graph(sess.graph_def,
                                 FLAGS.log_dir,
                                 "graph.pb",
                                 as_text=False)
            print("Loaded #tests:%d" % (len(testDatas)))
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    clipStep = int(step / 50000)
                    clipStep = clipStep * 50000
                    trainDatas = sess.run(batch_inputs)
                    lr = FLAGS.learning_rate - decayPerStep * clipStep
                    feedDict = make_feed_dict(
                        model, trainDatas, FLAGS.droprate)
                    trainLoss, accuracy, nspAcc, _ = sess.run(
                        [loss, acc_S, acc_S2, train_op], feedDict)
                    if (step + 1) % 100 == 0:
                        print("[%d] loss: [%r], accuracy:%.3f%%, nsp:%.3f%%" %
                              (step + 1, trainLoss, accuracy*100.0, nspAcc*100.0))
                    if (step + 1) % 5000 == 0 or step == 0:
                        tloss, tacc, tacc2, targetAcc = test_eval(
                            sess, loss_t, sameCountS_t, acc_S_t, acc_S2_t, testDatas,
                            FLAGS.test_batch_size, model)
                        print("test loss:%.3f, accuracy1:%.3f%%, accuracy2:%.3f%%, nsp:%.3f%%" %
                              (tloss, tacc*100.0, tacc2*100.0, targetAcc*100.0))
                        if step and tloss < bestLoss:
                            bestSaver.save(
                                sess, FLAGS.best_dir + '/best_model')
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
