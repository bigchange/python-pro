#!/usr/bin/env python
# -*- coding:utf-8 -*-
# <<licensetext>>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import numpy as np
import tensorflow as tf
import match_model_adjust as matchm
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', "data/train.tfrecord",
                           'Training data path')
tf.app.flags.DEFINE_string('test_data_path', "data/test.tfrecord",
                           'Test data path')
tf.app.flags.DEFINE_string('log_dir', "logs", 'The log  dir')
tf.app.flags.DEFINE_string("wordvec_path", "matching/corpus/vec.txt",
                           "the word word2vec data path")
tf.app.flags.DEFINE_integer("wordvec_size", 150, "the vec embedding size")
tf.app.flags.DEFINE_integer(
    "job_embedding_size", 30, "the job embedding  size")
tf.app.flags.DEFINE_integer("max_tokens_per_sentence", 120,
                            "max num of tokens per sentence")

tf.app.flags.DEFINE_integer("num_titles", 1930, "number of title")
tf.app.flags.DEFINE_integer("max_epochs", 100, "max num of epoches")

tf.app.flags.DEFINE_integer("batch_size", 64, "num example per mini batch")
tf.app.flags.DEFINE_integer("test_batch_size", 256,
                            "num example per test batch")
tf.app.flags.DEFINE_integer("train_steps", 500000, "trainning steps")
tf.app.flags.DEFINE_integer("track_history", 10, "track max history accuracy")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float("learning_rate_min", 0.00001,
                          "the final minimal learning rate")
tf.app.flags.DEFINE_string("job_embedding_path", "matching/corpus/title_embedding.txt",
                           "the job embedding data path")


def load_w2v(path, expectDim, checkUNK=True):
    fp = open(path, "r")
    print("load data from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == expectDim)
    ws = []
    mv = [0 for i in range(dim)]
    second = -1
    for t in range(total):
        if ss[0] == '<UNK>':
            second = t
        line = fp.readline().strip()
        ss = line.split(" ")
        assert (len(ss) == (dim + 1))
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)
    for i in range(dim):
        mv[i] = mv[i] / total
    assert((not checkUNK) or (second != -1))
    # append two more token , maybe useless
    ws.append(mv)
    ws.append(mv)
    if checkUNK and second != 1:
        t = ws[1]
        ws[1] = ws[second]
        ws[second] = t
    fp.close()
    print("loaded word2vec:%d" % (len(ws)))
    return np.asarray(ws, dtype=np.float32)


def test_eval(sess, loss, t1_S, t3_S, t10_S, testDatas, batchSize, model):
    totalLen = len(testDatas)
    numBatch = int((totalLen - 1) / batchSize) + 1
    totalLoss = 0
    t3 = 0
    t1 = 0
    t10 = 0
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        X = testDatas[i * batchSize:endOff]
        feed_dict = {model.label_target_holder: [x[0] for x in X],
                     model.targets_holder: [x[1] for x in X],
                     model.gender_holder: [x[2] for x in X],
                     model.age_holder: [x[3] for x in X],
                     model.location_holder: [x[4] for x in X],
                     model.education_schools_holder: [x[5] for x in X],
                     model.education_degrees_holder: [x[6] for x in X],
                     model.education_starts_holder: [x[7] for x in X],
                     model.education_majors_holder: [x[8] for x in X],
                     model.work_expr_orgs_holder: [x[9] for x in X],
                     model.work_expr_starts_holder: [x[10] for x in X],
                     model.work_expr_durations_holder: [x[11] for x in X],
                     model.work_expr_jobs_holder: [x[12] for x in X],
                     model.work_expr_orgIds_holder:[x[13] for x in X],
                     model.work_expr_descs_holder: [x[14] for x in X],
                     model.proj_expr_descs_holder: [x[15] for x in X], }
        lossv, t1v, t3v, t10v = sess.run([loss, t1_S, t3_S, t10_S], feed_dict)
        totalLoss += lossv
        t3 += t3v
        t1 += t1v
        t10 += t10v
    return totalLoss / numBatch, float(t1) / totalLen, float(
        t3) / totalLen, float(t10) / totalLen


def load_test_dataset_all(sess, test_input, testDatas):
    while True:
        try:
            target, targets, gender, age, location, education_schools, education_degrees, education_starts, education_majors, work_expr_orgs, work_expr_starts, work_expr_durations, work_expr_jobs, work_expr_orgIds, work_expr_descs, proj_expr_descs = sess.run(
                test_input)
            testDatas.append(
                [target, targets, gender, age, location, education_schools,
                 education_degrees, education_starts, education_majors,
                 work_expr_orgs, work_expr_starts, work_expr_durations,
                 work_expr_jobs, work_expr_orgIds, work_expr_descs, proj_expr_descs])
        except tf.errors.OutOfRangeError:
            break


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
        datasetTrain = tf.contrib.data.TFRecordDataset(trainDataPaths)
        datasetTrain = datasetTrain.map(matchm.parse_tfrecord_function)
        datasetTrain = datasetTrain.repeat(FLAGS.max_epochs)
        datasetTrain = datasetTrain.shuffle(buffer_size=2000)
        datasetTrain = datasetTrain.batch(FLAGS.batch_size)
        iterator = datasetTrain.make_one_shot_iterator()
        batch_inputs = iterator.get_next()

        datasetTest = tf.contrib.data.TFRecordDataset(testDataPaths)
        datasetTest = datasetTest.map(matchm.parse_tfrecord_function)

        iteratorTest = datasetTest.make_initializable_iterator()
        test_input = iteratorTest.get_next()
        wordsEm = load_w2v(FLAGS.wordvec_path, FLAGS.wordvec_size)
        jobsEm = None
        if FLAGS.job_embedding_path:
            jobsEm = load_w2v(FLAGS.job_embedding_path,
                              FLAGS.job_embedding_size, checkUNK=False)
        model = matchm.MatchModel(
            wordsEm, num_class=FLAGS.num_titles, jobEmbeddings=jobsEm)
        print("train data path:%s" % (",".join(trainDataPaths)))
        loss, _, _, _ = model.loss(batch_inputs)
        testLoss, t1_S, t3_S, t10_S = model.test_loss()
        train_op = model.train(loss)
        col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for x in col:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, x)
        decayPerStep = (
            FLAGS.learning_rate - FLAGS.learning_rate_min) / FLAGS.train_steps
        sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
        with sv.managed_session(master='') as sess:
            # actual training loop
            training_steps = FLAGS.train_steps
            bestLoss = float("inf")
            trackHist = 0
            sess.run([iteratorTest.initializer])
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
                    clipStep = int(step / 20000)
                    clipStep = clipStep * 20000
                    trainLoss, _ = sess.run(
                        [loss, train_op], {model.learning_rate_h: (
                            FLAGS.learning_rate - decayPerStep * clipStep)})
                    if (step + 1) % 100 == 0:
                        print("[%d] loss: [%r]" % (step + 1, trainLoss))
                    if (step + 1) % 2000 == 0 or step == 0:
                        tloss, acc1, acc3, acc10 = test_eval(
                            sess, testLoss, t1_S, t3_S, t10_S, testDatas,
                            FLAGS.test_batch_size, model)

                        print(
                            "test loss:%.3f, top1 acc:%.3f,top3 acc:%.3f,top10 acc:%.3f"
                            % (tloss, acc1, acc3, acc10))
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
