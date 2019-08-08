# -*- coding: utf-8 -*-
# @Date:   2019-06-11 16:21:27
# @Last Modified time: 2019-06-12 11:37:59
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import numpy as np
import tensorflow as tf
import model_industry as model
import random
import optimization
import model_transfer as mtransfer


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_data_path', "/data/resume_jd/e2e/data/train2.tf",
                           'Training data path')
tf.app.flags.DEFINE_string('test_data_path', "/data/resume_jd/e2e/data/test2.tf",
                           'Test data path')

tf.app.flags.DEFINE_string('model_log_dir', "logs_industry_cls", 'The log  dir')

tf.app.flags.DEFINE_string('summaries_dir', "summary",
                           'summary  path')


tf.app.flags.DEFINE_string('pretrained', '', 'the pretrained model dir')

tf.app.flags.DEFINE_integer("max_epochs", 2000, "max num of epoches")

tf.app.flags.DEFINE_integer("batch_size", 64, "num example per mini batch")
tf.app.flags.DEFINE_integer("step_size", 350000, "num example per mini batch")
tf.app.flags.DEFINE_integer("warm_steps", 3000, "warm steps for training")
tf.app.flags.DEFINE_integer("test_batch_size", 6,
                            "num example per test batch")
tf.app.flags.DEFINE_integer("train_steps", 50000, "trainning steps")
tf.app.flags.DEFINE_integer("track_history", 20, "track max history accuracy")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float("dropout_rate", 0.3, "the dropout rate")

tf.app.flags.DEFINE_integer("tag_size", 31,
                            "num example per test batch")


def parse_tfrecord_function(example_proto):
    features = {
        "sentence": tf.FixedLenSequenceFeature([],
                                               tf.int64,
                                               allow_missing=True,
                                               default_value=0),
        "label_targets": tf.FixedLenSequenceFeature([], tf.float32,
                                            allow_missing=True,
                                            default_value=0.0),
        "a_length": tf.FixedLenFeature([], tf.int64,
                                     default_value=0),
        "ab_length": tf.FixedLenFeature([], tf.int64,
                                     default_value=0)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features


def make_feed_dict(model,
                   one_batch_feature_map,
                   droprate=0):
    dicts = {model.targets: one_batch_feature_map["label_targets"],
             model.sentence_holder: one_batch_feature_map["sentence"],
             model.check_alen_holder: one_batch_feature_map["a_length"],
             model.check_tlen_holder: one_batch_feature_map["ab_length"],
             model.dropout_h: droprate}
    return dicts


def test_eval(sess, model, loss,
              test_input_feature_map):
    numBatch = 0
    totalLoss = 0
    while True:
        try:
            one_batch_feature_map = sess.run(test_input_feature_map)
            feed_dict = make_feed_dict(model, one_batch_feature_map)

            lossv = sess.run(loss, feed_dict)

            totalLoss += lossv
            numBatch += 1
            if numBatch == 20:
                break
        except tf.errors.OutOfRangeError:
            break
    totalLoss /= numBatch
    print("task test loss:%.3f" %
          (totalLoss))
    return totalLoss


def make_input(dataPath ,parseFunc, is_trainning, maxEpochs=None, batchSize=None):
    dataset = tf.data.TFRecordDataset(
          dataPath, compression_type='GZIP', buffer_size=4096*1024*10)
    dataset = dataset.map(
              parseFunc)
    if batchSize is not None:
      dataset = dataset.batch(batchSize)
    if is_trainning:
        if maxEpochs is not None:
          dataset = dataset.repeat(maxEpochs)
          dataset = dataset.shuffle(buffer_size=800)

        iterator = dataset.make_one_shot_iterator()
        batch_inputs_feature_map = iterator.get_next()
        return batch_inputs_feature_map
    else:
        iterator = dataset.make_one_shot_iterator()
        batch_inputs_feature_map = iterator.get_next()

        return batch_inputs_feature_map

def cross_entropy_loss(preds, targets):
      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          labels=targets, logits=preds))
      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      cost += tf.reduce_sum(reg_losses)
      #tf.summary.scalar('cross_entropy_loss', cost)
      return cost

def multi_label_loss(preds, targets):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                          labels=targets, logits=preds))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cost += tf.reduce_sum(reg_losses)
    #tf.summary.scalar('cross_entropy_loss', cost)
    return cost

def multi_label_loss2(preds, targets):
    cost = tf.nn.sigmoid_cross_entropy_with_logits(
                          labels=targets, logits=preds)
    cost = tf.reduce_sum(cost, 1)
    cost = tf.reduce_mean(cost)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cost += tf.reduce_sum(reg_losses)
    #tf.summary.scalar('cross_entropy_loss', cost)
    return cost

def main(unused_argv):
    curdir = os.path.dirname(os.path.realpath(__file__))
    trainDataPaths = tf.app.flags.FLAGS.train_data_path.split(",")

    testDataPaths = tf.app.flags.FLAGS.test_data_path.split(",")

    with tf.Graph().as_default():
        m = model.Model(FLAGS.tag_size)
        batch_inputs_feature_map = make_input(trainDataPaths,
                                             parse_tfrecord_function,
                                              True,
                                              FLAGS.max_epochs,
                                              FLAGS.batch_size)
        test_input_feature_map = make_input(testDataPaths,
                                            parse_tfrecord_function,
                                            False,
                                            None,
                                            FLAGS.test_batch_size)


        partialSaver = None
        if FLAGS.pretrained:
            partialSaver = mtransfer.partial_transfer(FLAGS.pretrained)
        loss = multi_label_loss2(m.preds, m.targets)
        lrate = 1.0
        train_op = optimization.create_optimizer(loss, FLAGS.learning_rate,
                                                 FLAGS.train_steps, FLAGS.warm_steps, False, lr_decrease_map={b'tf_inference':lrate})
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sv = tf.train.Supervisor(logdir=FLAGS.model_log_dir, save_model_secs=3600)
        with sv.managed_session(master='') as sess:
            # actual training loop
            training_steps = FLAGS.train_steps
            best_loss = float("inf")
            trackHist = 0

            if partialSaver:
                partialSaver.restore(sess, FLAGS.pretrained)

            tf.train.write_graph(sess.graph_def,
                                 FLAGS.model_log_dir,
                                 "graph.pb",
                                 as_text=False)
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    one_batch_feature_map = sess.run(batch_inputs_feature_map)
                    feedDict = make_feed_dict(
                        m, one_batch_feature_map, FLAGS.dropout_rate)
                    trainLoss, _ = sess.run(
                        [loss, train_op], feedDict)
                    if (step + 1) % 50 == 0:
                        print("[%d] loss: [%r]" %
                              (step + 1, trainLoss))
                    if (step + 1) % 1000 == 0 or step == 0:
                        tloss = test_eval(sess, m, loss,
                                          test_input_feature_map)
                        if step and tloss < best_loss:
                            sv.saver.save(
                                sess, FLAGS.model_log_dir + '/best_model')
                            trackHist = 0
                            best_loss = tloss
                            print("save best loss :%.3f%%" % (tloss))
                        else:
                            if trackHist >= FLAGS.track_history:
                                print(
                                    "always not good enough in last %d histories, best loss:%.3f"
                                    % (trackHist, best_loss))
                                break
                            else:
                                trackHist += 1

                except KeyboardInterrupt as e:
                    sv.saver.save(sess,
                                  FLAGS.model_log_dir + '/model',
                                  global_step=(step + 1))
                    raise e
            sv.saver.save(sess, FLAGS.model_log_dir + '/finnal-model')

if __name__ == '__main__':
    tf.app.run()