#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: train_chatbot.py
# Project: py

# Copyright 2020 - 2019


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import numpy as np
import random
import tensorflow as tf
from summary_model import SummaryModel as Model
from optimization import create_optimizer
import model_transfer as mtransfer

tf.app.flags.DEFINE_string('train_data_path', "/data/chatbot/tf/train.tfrecord",
                           'Training data path')
tf.app.flags.DEFINE_string('test_data_path', "/data/chatbot/tf/test.tfrecord",
                           'Test data path')
tf.app.flags.DEFINE_string('log_dir', "chat_logs", 'The log  dir')
tf.app.flags.DEFINE_integer(
    "embedding_size", 200, "the embedding  size")
tf.app.flags.DEFINE_integer(
    "max_length", 299, "the max input  length")
tf.app.flags.DEFINE_integer(
    "max_y_length", 127, "the max output  length")
tf.app.flags.DEFINE_integer(
    "vocab_size", 32003, "the vocab size")
tf.app.flags.DEFINE_integer("max_epochs", 100, "max num of epoches")
tf.app.flags.DEFINE_integer("batch_size", 256, "num example per mini batch")
tf.app.flags.DEFINE_integer("test_batch_size", 256,
                            "num example per test batch")
tf.app.flags.DEFINE_integer("train_steps", 180000, "trainning steps")
tf.app.flags.DEFINE_integer("track_history", 6, "track max history accuracy")
tf.app.flags.DEFINE_integer("warm_steps", 20000, "track max history accuracy")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_float("dropout_rate", 0.2, "dropout rate")
tf.app.flags.DEFINE_string('pretrained', "",
                           'the  pre-trained  bert path')

FLAGS = tf.app.flags.FLAGS


def parse_tfrecord_function(example_proto):
    features = {
        "y": tf.FixedLenSequenceFeature([], tf.int64,
                                        allow_missing=True,
                                        default_value=0),
        "x": tf.FixedLenSequenceFeature([], tf.int64,
                                        allow_missing=True,
                                        default_value=0),
        "xLen": tf.FixedLenFeature([], tf.int64,
                                   default_value=0),
        "yLen": tf.FixedLenFeature([], tf.int64,
                                   default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    x = parsed_features["x"]
    x.set_shape([FLAGS.max_length])
    y = parsed_features["y"]
    y.set_shape([FLAGS.max_y_length])
    xlen = parsed_features["xLen"]
    ylen = parsed_features["yLen"]
    return x, y, xlen+1, ylen


def test_eval(sess, loss, same_S, total_S, testDatas, batchSize, preds, model):
    totalLen = len(testDatas)
    numBatch = int((totalLen - 1) / batchSize) + 1
    totalLoss = 0
    same = 0
    total = 0
    predsv = None
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        tDatas = testDatas[i * batchSize:endOff]
        tX = [x[0] for x in tDatas]
        tY = [x[1] for x in tDatas]
        tXL = [x[2] for x in tDatas]
        tYL = [x[3] for x in tDatas]
        mdlen = np.max(tYL) + 1
        if mdlen > FLAGS.max_length:
            mdlen = FLAGS.max_length

        feed_dict = {model.inp_X:  tX,
                     model.inp_Y:  tY,
                     model.a_length: tXL,
                     model.totalLength: tXL,
                     model.target_length: tYL,
                     model.max_decode_length: mdlen}
        lossv, samev, totalv = 0, 0, 0
        if predsv is None and random.random() < 0.1:
            lossv, samev, totalv, predsv = sess.run(
                [loss, same_S, total_S, preds], feed_dict)
            predsv = predsv[0]
        else:
            lossv, samev, totalv = sess.run(
                [loss, same_S, total_S], feed_dict)
        totalLoss += lossv
        same += samev
        total += totalv

    if predsv is not None:
        ss = [str(v) for v in predsv]
        print("predicts len:%d,content is :[%s]" % (len(predsv), " ".join(ss)))
    print("total:%d, diff:%d" % (total, same))
    # accuracy = 100.0 * correct_labels / float(total_labels)
    return 1.0-same / (total+0.0000001), totalLoss / numBatch


def load_test_dataset_all(sess, test_input, testDatas):
    loaded = 0
    while True:
        try:
            datas = sess.run(
                test_input)
            testDatas.append(
                datas)
            loaded += 1
            if loaded >= 8000:
                break
        except:
            break


def make_feed_dict(model,
                   inputs,
                   droprate=0,
                   lr=None,
                   no_pair=True):
    mdlen = np.max(inputs[3])+1
    if mdlen > FLAGS.max_length:
        mdlen = FLAGS.max_length
    dicts = {model.inp_X: inputs[0],
             model.inp_Y: inputs[1],
             model.a_length: inputs[2],
             model.totalLength: inputs[2],
             model.target_length: inputs[3],
             model.max_decode_length: mdlen}
    return dicts


def main(unused_argv):
    curdir = os.path.dirname(os.path.realpath(__file__))
    trainDataPaths = tf.app.flags.FLAGS.train_data_path.split(",")
    for i in range(len(trainDataPaths)):
        if not trainDataPaths[i].startswith("/"):
            trainDataPaths[i] = curdir + "/" + trainDataPaths[i]

    testDataPaths = tf.app.flags.FLAGS.test_data_path.split(",")
    for i in range(len(testDataPaths)):
        if not testDataPaths[i].startswith("/"):
            testDataPaths[i] = curdir + "/" + testDataPaths[i]

    graph = tf.Graph()
    testDatas = []
    with graph.as_default():
        datasetTrain = tf.data.TFRecordDataset(
            trainDataPaths)
        datasetTrain = datasetTrain.map(parse_tfrecord_function)
        datasetTrain = datasetTrain.repeat(FLAGS.max_epochs)
        datasetTrain = datasetTrain.shuffle(buffer_size=2048)
        datasetTrain = datasetTrain.batch(FLAGS.batch_size)
        iterator = datasetTrain.make_one_shot_iterator()
        batch_inputs = iterator.get_next()

        datasetTest = tf.data.TFRecordDataset(
            testDataPaths)
        datasetTest = datasetTest.map(parse_tfrecord_function)

        iteratorTest = datasetTest.make_initializable_iterator()
        test_input = iteratorTest.get_next()
        model = Model(FLAGS.max_length, FLAGS.max_y_length)
        print("train data path:%s" % (",".join(trainDataPaths)))
        loss, _, _, _ = model.loss(training=True)
        train_op = create_optimizer(
            loss, FLAGS.learning_rate, FLAGS.train_steps, FLAGS.warm_steps, False)
        loss_t, same_s,total_s,pred_s = model.loss(training=False)
        partialSaver = None
        if FLAGS.pretrained:
            partialSaver = mtransfer.partial_transfer(FLAGS.pretrained)
        sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
        with sv.managed_session(master='') as sess:
            # actual training loop
            training_steps = FLAGS.train_steps
            bestLoss = float("inf")
            bestAcc = 0
            trackHist = 0
            if partialSaver:
                partialSaver.restore(sess, FLAGS.pretrained)
            sess.run(iteratorTest.initializer)
            load_test_dataset_all(sess, test_input, testDatas)
            tf.train.write_graph(sess.graph_def,
                                 FLAGS.log_dir,
                                 "graph.pb",
                                 as_text=False)
            print("Loaded #tests:%d" % (len(testDatas)))
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    trainDatas = sess.run(batch_inputs)
                    feedDict = make_feed_dict(
                        model, trainDatas, FLAGS.dropout_rate)
                    trainLoss, _ = sess.run(
                        [loss, train_op], feedDict)
                    if (step + 1) % 100 == 0:
                        print("[%d] loss: [%r]" % (step + 1, trainLoss))
                    if (step + 1) % 1000 == 0 or step == 0:
                        accuracy, tloss = test_eval(
                            sess, loss_t, same_s, total_s, testDatas,
                            FLAGS.test_batch_size, pred_s, model)
                        print(
                            "test loss:%.3f, token acc:%.3f"
                            % (tloss, accuracy))
                        if step and tloss < bestLoss:
                            sv.saver.save(
                                sess, FLAGS.log_dir + '/best_model')
                            trackHist = 0
                            bestLoss = tloss
                            bestAcc = accuracy
                        else:
                            if trackHist >= FLAGS.track_history:
                                print(
                                    "always not good enough in last %d histories, best loss:%.3f(%.3f%%)"
                                    % (trackHist, bestLoss, bestAcc * 100))
                                break
                            else:
                                trackHist += 1
                except KeyboardInterrupt as e:
                    sv.saver.save(sess,
                                  FLAGS.log_dir + '/model',
                                  global_step=(step + 1))
                    raise e
            sv.saver.save(sess, FLAGS.log_dir + '/finnal-model')


if __name__ == '__main__':
    tf.app.run()
