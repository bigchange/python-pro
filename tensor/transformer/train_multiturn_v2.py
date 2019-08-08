#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: train_multiturn_v2.py
# Project: transformer

# Copyright 2020 - 2019


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import numpy as np
import random
import tensorflow as tf
from multiturn_model_v2 import MultiTurnModelV2 as Model
from optimization import create_optimizer
import model_transfer as mtransfer
import eval_douban as doubanDataset

tf.app.flags.DEFINE_string('train_data_path', "/data/multiturn/tf/train.tfrecord",
                           'Training data path')
tf.app.flags.DEFINE_string('test_data_path', "/data/multiturn/tf/test.tfrecord",
                           'Test data path')
tf.app.flags.DEFINE_string('dev_data_path', "/data/multiturn/tf/dev.tfrecord",
                           'Dev data path')
tf.app.flags.DEFINE_string('log_dir', "multiturn_logs_v2", 'The log  dir')
tf.app.flags.DEFINE_integer(
    "embedding_size", 200, "the embedding  size")
tf.app.flags.DEFINE_integer(
    "max_length", 95, "the max input  length")
tf.app.flags.DEFINE_integer(
    "max_hist", 9, "the max history num")
tf.app.flags.DEFINE_integer(
    "vocab_size", 32003, "the vocab size")
tf.app.flags.DEFINE_integer("max_epochs", 100, "max num of epoches")
tf.app.flags.DEFINE_integer("batch_size", 64, "num example per mini batch")
tf.app.flags.DEFINE_integer("test_batch_size", 64,
                            "num example per test batch")
tf.app.flags.DEFINE_integer("train_steps", 480000, "trainning steps")
tf.app.flags.DEFINE_integer("track_history", 8, "track max history accuracy")
tf.app.flags.DEFINE_integer("warm_steps", 30000, "track max history accuracy")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_float("dropout_rate", 0.2, "dropout rate")
tf.app.flags.DEFINE_string('pretrained', "",
                           'the  pre-trained  bert path')

FLAGS = tf.app.flags.FLAGS


def parse_tfrecord_function(example_proto):
    features = {
        "types": tf.FixedLenSequenceFeature([], tf.int64,
                                        allow_missing=True,
                                        default_value=0),
        "allIds": tf.FixedLenSequenceFeature([], tf.int64,
                                        allow_missing=True,
                                        default_value=0),
        "aLen": tf.FixedLenSequenceFeature([], tf.int64,
                                    allow_missing=True,
                                   default_value=0),
        "tLen": tf.FixedLenSequenceFeature([], tf.int64,
                                   allow_missing=True,
                                   default_value=0),
        "target": tf.FixedLenFeature([], tf.int64,
                                   default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    x = parsed_features["allIds"]
    x.set_shape([FLAGS.max_length*FLAGS.max_hist])
    x=tf.reshape(x,[FLAGS.max_hist,FLAGS.max_length])
    types = parsed_features["types"]
    types.set_shape([FLAGS.max_hist])
    alens = parsed_features["aLen"]
    alens.set_shape([FLAGS.max_hist])
    tlens = parsed_features["tLen"]
    tlens.set_shape([FLAGS.max_hist])
    target = parsed_features["target"]
    return x, types, alens, tlens, target

def do_eval_test(sess, pred_s, testDatas, batchSize, model, outpath):
    totalLen = len(testDatas)
    numBatch = int((totalLen - 1) / batchSize) + 1
    totalLoss = 0
    same = 0
    total = 0
    with open(outpath,"w") as outp:
        for i in range(numBatch):
            endOff = (i + 1) * batchSize
            if endOff > totalLen:
                endOff = totalLen
            tDatas = testDatas[i * batchSize:endOff]
            total += len(tDatas)
            tX = [x[0] for x in tDatas]
            tP = [x[1] for x in tDatas]
            tAL = [x[2] for x in tDatas]
            tTL = [x[3] for x in tDatas]
            tY = [x[4] for x in tDatas]
            feed_dict = {model.inp_X:  tX,
                        model.y:  tY,
                        model.a_length: tAL,
                        model.t_length: tTL,
                        model.seg_types: tP}
            preds = sess.run(
                    [pred_s], feed_dict)
            preds = preds[0]
            # print("got #%d preds"%(len(preds)))
            for i in range(len(preds)):
                outp.write("%.5f\t%d\n"%(preds[i], tY[i]))
    
    _,_,_,r_1,_,_=doubanDataset.evaluate(outpath)
    return r_1

def dev_eval(sess, loss, same_S, testDatas, batchSize, model):
    totalLen = len(testDatas)
    numBatch = int((totalLen - 1) / batchSize) + 1
    totalLoss = 0
    same = 0
    total = 0
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        tDatas = testDatas[i * batchSize:endOff]
        total += len(tDatas)
        tX = [x[0] for x in tDatas]
        tP = [x[1] for x in tDatas]
        tAL = [x[2] for x in tDatas]
        tTL = [x[3] for x in tDatas]
        tY = [x[4] for x in tDatas]
        

        feed_dict = {model.inp_X:  tX,
                     model.y:  tY,
                     model.a_length: tAL,
                     model.t_length: tTL,
                     model.seg_types: tP}        
        lossv, samev = sess.run(
                [loss, same_S], feed_dict)
        totalLoss += lossv
        same += samev

    print("total:%d, same:%d" % (total, same))
    # accuracy = 100.0 * correct_labels / float(total_labels)
    return same / (total+0.0000001), totalLoss / numBatch


def load_test_dataset_all(sess, test_input, testDatas):
    loaded = 0
    while True:
        try:
            datas = sess.run(
                test_input)
            testDatas.append(
                datas)
            loaded += 1
        except:
            break


def make_feed_dict(model,
                   inputs,
                   droprate=0,
                   lr=None,
                   no_pair=True):
    dicts = {model.inp_X: inputs[0],
             model.seg_types: inputs[1],
             model.a_length: inputs[2],
             model.t_length: inputs[3],
             model.y: inputs[4]}
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
    
    devDataPaths = tf.app.flags.FLAGS.dev_data_path.split(",")
    for i in range(len(devDataPaths)):
        if not devDataPaths[i].startswith("/"):
            devDataPaths[i] = curdir + "/" + devDataPaths[i]

    graph = tf.Graph()
    testDatas = []
    devDatas =[]
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
        
        datasetDev = tf.data.TFRecordDataset(
            devDataPaths)
        datasetDev = datasetDev.map(parse_tfrecord_function)

        iteratorDev = datasetDev.make_initializable_iterator()
        dev_input = iteratorDev.get_next()
        
        model = Model(FLAGS.max_length)
        print("train data path:%s" % (",".join(trainDataPaths)))
        loss, same,_ = model.loss(training=True)
        train_op = create_optimizer(
            loss, FLAGS.learning_rate, FLAGS.train_steps, FLAGS.warm_steps, False)
        loss_t, same_s, pred_s= model.loss(training=False)
        partialSaver = None
        if FLAGS.pretrained:
            partialSaver = mtransfer.partial_transfer(FLAGS.pretrained)
        sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir, save_model_secs=3600*3)
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
            sess.run(iteratorDev.initializer)
            load_test_dataset_all(sess, dev_input, devDatas)
            tf.train.write_graph(sess.graph_def,
                                 FLAGS.log_dir,
                                 "graph.pb",
                                 as_text=False)
            print("Loaded #tests:%d" % (len(testDatas)))
            print("Loaded #devs:%d" % (len(devDatas)))
            for step in range(training_steps):
                if sv.should_stop():
                    break
                # do_eval_test(sess, pred_s, testDatas, FLAGS.test_batch_size, model, './eval_'+str(step)+".txt")
                try:
                    trainDatas = sess.run(batch_inputs)
                    feedDict = make_feed_dict(
                        model, trainDatas, FLAGS.dropout_rate)
                    trainLoss,trainSame, _ = sess.run(
                        [loss,same, train_op], feedDict)
                    if (step + 1) % 100 == 0:
                        print("[%d] loss: [%r] , acc=%f%%" % (step + 1, trainLoss,(trainSame*100.0)/(0.000001+FLAGS.batch_size) ))
                    if (step + 1) % 4000 == 0 or step == 0:
                        accuracy, tloss = dev_eval(
                            sess, loss_t, same_s, devDatas,
                            FLAGS.test_batch_size, model)
                        print(
                            "test loss:%.3f, token acc:%.3f"
                            % (tloss, accuracy))
                        print("===================now do eval on test...................")
                        r_1=do_eval_test(sess, pred_s, testDatas, FLAGS.test_batch_size, model, './eval_'+str(step)+".txt")
                        if step and r_1 > bestAcc:
                            sv.saver.save(
                                sess, FLAGS.log_dir + '/best_model'+str(step))
                            trackHist = 0
                            bestLoss = tloss
                            bestAcc = r_1
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
