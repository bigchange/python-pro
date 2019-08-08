#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: train_ner.py

# Copyright 2020 - 2018


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import math
import numpy as np
import random

import tensorflow as tf
import argparse

import json
from time import ctime

from optimization import create_optimizer
from bert_ner import parse_tfrecord_function, Model as NerModel
import model_transfer as mtransfer

# 训练命令：
# python jd_extractor/model/lattic_ner/train.py --word_vector_file matching/corpus/v120k_big.txt  --tag_vocab_file jd_extractor/model/cc/keyword.txt
# 模型导出命令
# freeze_graph --input_graph graph.pbtxt --input_checkpoint best_model --output_node_names finalInference,transitions   --output_graph jd_model.pb —input_binary

def parse_args():
    parser = argparse.ArgumentParser(description=' jd extractor ner model')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--droprate', default=0.3, type=float)
    parser.add_argument('--iternum', default=10000, type=int)
    parser.add_argument('--phase', default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--output', default='./out')
    parser.add_argument('--pretrained', default='')
    parser.add_argument('--dump_words', default='')
    parser.add_argument('--tag_vocab_file', default=None)
    parser.add_argument('--hidden_size', default=200, type=int)
    parser.add_argument('--embedding_size', default=180, type=int)

    parser.add_argument('--weights', default=None)
    parser.add_argument(
        '--tfdata', default='/e/code/idmg/jd_extractor/model/train')
    parser.add_argument('--log_dir', default='./log_jdner')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--track_history', default=100)
    parser.add_argument('--warm_steps', default=1000)
    parser.add_argument('--train_steps', default=100000)
    return parser.parse_args()


args = parse_args()


def make_feed_dict(model, inputs, droprate=0):
    return {
        model.sentence_placeholder: inputs[0],
        model.targets: inputs[1],
        model.a_length: inputs[2],
        model.dropout_h: droprate
    }


def load_test_dataset(sess, test_input, testDatas):
    while True:
        try:
            features = sess.run(test_input)
            testDatas.append(features)
        except:
            break
    print("loaded %d test datas" % (len(testDatas)))


def test_eval(sess, unary_score, test_squence_length,
              transMatrix, model, testDatas, loss):
    batch = args.batch_size
    len_test = len(testDatas)
    numbatch = int((len_test-1)/args.batch_size)
    correct_labels = 0
    total_labels = 0
    total_loss = 0.0
    total_num = 0
    for i in range(numbatch):
        endOff = (i+1)*batch
        if endOff > len_test:
            endOff = len_test
        data = testDatas[i*batch:endOff]
        total_num += len(data)
        sentence = [b[0] for b in data]
        targets = [b[1] for b in data]
        lengths = [b[2] for b in data]

        inputs = (sentence, targets, lengths)

        feed_dict = make_feed_dict(model, inputs)
        y = inputs[1]
        unary_score_val, length, b_loss = sess.run(
            [unary_score, test_squence_length, loss], feed_dict
        )
        total_loss += b_loss * len(data)
        for unary_, y_, l_ in zip(unary_score_val, y, length):
            unary_ = unary_[:l_-1]
            y_ = y_[:l_-1]
            viterbi_s, _ = tf.contrib.crf.viterbi_decode(unary_, transMatrix)
            correct_labels += np.sum(np.equal(viterbi_s, y_))
            total_labels += l_-1
    accuracy = 100.0*correct_labels/float(total_labels)
    print('Accuracy: %f, loss: %f' % (accuracy, total_loss / total_num))
    return accuracy

def _get_tags_from_files(in_tag_file, vocab):
    with open(in_tag_file) as f:
        for line in f.readlines():
            if line is None:
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            ss = line.split()
            if len(ss) != 2:
                continue
            vocab[ss[0].decode('utf8')] = int(ss[1])
        print('length of vocab is ' + str(len(vocab)))


def _get_tags_all(in_vocab_tag, out_vocab_tag_all):
    for key, val in in_vocab_tag.items():
        out_vocab_tag_all[key+'-b'] = 2*val
        out_vocab_tag_all[key+'-i'] = 2*val-1
    out_vocab_tag_all[u'other'] = len(in_vocab_tag)*2+1


def main(args):
    tfdatapath = args.tfdata
    train_file = os.path.join(tfdatapath, 'tfrecord.train')
    test_file = os.path.join(tfdatapath, 'tfrecord.test')
    graph = tf.Graph()
    testDatas = []
    with graph.as_default():
        datasetTrain = tf.data.TFRecordDataset(train_file)
        datasetTrain = datasetTrain.map(parse_tfrecord_function)
        datasetTrain = datasetTrain.repeat(args.iternum)
        datasetTrain = datasetTrain.shuffle(buffer_size=1024)
        datasetTrain = datasetTrain.batch(args.batch_size)
        iterator = datasetTrain.make_one_shot_iterator()
        batch_inputs = iterator.get_next()
        datasetTest = tf.data.TFRecordDataset(test_file)
        datasetTest = datasetTest.map(parse_tfrecord_function)
        iteratorTest = datasetTest.make_initializable_iterator()
        test_input = iteratorTest.get_next()

        # 2. load tag vocab
        vocab_tags = {}
        _get_tags_from_files(args.tag_vocab_file, vocab_tags)
        vocab_tags_all = {}
        _get_tags_all(vocab_tags, vocab_tags_all)

        # 4. build model
        print('====> building ner model: ')
        model = NerModel(
            args.embedding_size,
            len(vocab_tags_all)+1)
        pred, pLen = model.inference()
        loss = model.loss(pred, pLen)
        train_op = create_optimizer(
            loss, args.lr, args.train_steps, args.warm_steps, False)
        partialSaver = None
        if args.pretrained:
            partialSaver = mtransfer.partial_transfer(args.pretrained)
        sv = tf.train.Supervisor(graph=graph, logdir=args.log_dir)
        with sv.managed_session(master='') as sess:
            sess.run(iteratorTest.initializer)
            load_test_dataset(sess, test_input, testDatas)
            bestAcc = -float('inf')
            trackHist = 0
            if partialSaver:
                partialSaver.restore(sess, args.pretrained)
            steps = 100000
            if args.dump_words:
                words=sess.run(model.words)
                nn=len(words)
                print("got %d words!"%(nn))
                maxV = -1000000
                minV = 1000000
                with open(args.dump_words,"w") as fout:
                    for i in range(nn):
                        ls =[]
                        for fv in words[i]:
                            if fv > maxV:
                                maxV = fv
                            if fv < minV:
                                minV = fv
                            ls.append(str(fv))
                        fout.write("%d %s\n"%(i, " ".join(ls)))
                print("dump words done, maxV = %f, minV=%f"%(maxV, minV))
            for i in range(steps):
                if sv.should_stop():
                    break
                try:
                    inputs = sess.run(batch_inputs)
                    feeddict = make_feed_dict(
                        model,
                        inputs,
                        droprate=args.droprate
                    )
                    trainLoss, transMatrix, _ = sess.run(
                        [loss, model.transition_params, train_op], feeddict)
                    if (i+1) % 50 == 0:
                        print('%s %d  loss: %f' % (ctime(), i+1, trainLoss))
                    if (i+1) % 100 == 0:
                        acc = test_eval(sess,
                                        pred, pLen, transMatrix,
                                        model, testDatas, loss)
                        if acc > bestAcc:
                            print(
                                '====> Current best accuracy: {:.3f}'.format(acc))
                            bestAcc = acc
                            trackHist = 0
                            sv.saver.save(sess, args.log_dir + '/best_model')
                        else:
                            if trackHist > args.track_history:
                                print('====> Alaways not better in last {} histories. '
                                      'Best Accuracy: {:.3f}'.format(trackHist, bestAcc))
                                break
                            else:
                                trackHist += 1
                except KeyboardInterrupt as e:
                    sv.saver.save(
                        sess, args.log_dir + '/model', global_step=(i + 1))
                    raise e
                except Exception as e:
                    print(e)
                    continue
            sv.saver.save(sess, args.log_dir + '/finnal-model')


if __name__ == '__main__':
    main(args)
