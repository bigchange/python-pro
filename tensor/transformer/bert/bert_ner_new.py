#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: bert_ner.py
# Project: bert

# Copyright 2020 - 2018


import tensorflow as tf
import numpy as np
import math

import bert_official as modeling
from bert_official import BertConfig, BertModel

kMaxSeqLen = 128
kMaxSubToken = 3
kMaxTarget = 3
kVocabSize = 50003


def parse_tfrecord_function(example_proto):
    features = {
        "targets":
        tf.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True, default_value=0),
        "sentence":
        tf.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True, default_value=0),
        "types":
        tf.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True, default_value=0),
        "length":
        tf.FixedLenFeature([], tf.int64, default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    sentence = parsed_features["sentence"]
    sentence.set_shape([kMaxSeqLen-1])
    types = parsed_features["types"]
    types.set_shape([kMaxSeqLen-1])
    targets = parsed_features["targets"]
    targets.set_shape([kMaxSeqLen-1])
    length = parsed_features["length"]+1
    return sentence, targets, length, types


class Model(object):
    def __init__(self,
                 embeddingSize,
                 numTags,
                 lastHiddenSize=200):
        self.max_token_per_sentence = kMaxSeqLen-1
        self.embedding_size = embeddingSize
        self.sentence_placeholder = tf.placeholder(
            tf.int32,
            shape=[None, self.max_token_per_sentence],
            name="sentence")
        self.types_placeholder = tf.placeholder(
            tf.int32,
            shape=[None, self.max_token_per_sentence],
            name="types")
        self.targets = tf.placeholder(
            tf.int32, shape=[None, self.max_token_per_sentence])
        self.a_length = tf.placeholder(tf.int32, shape=[None], name="alength")
        self.totalLength = tf.placeholder(tf.int32, shape=[None], name="totalLength")

        self.bert_config = BertConfig(
            kVocabSize, hidden_size=200, num_hidden_layers=6, num_attention_heads=8, intermediate_size=800)
        self.last_hidden_size = lastHiddenSize
        self.learning_rate_h = tf.placeholder(tf.float32, shape=[])
        self.dropout_h = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.num_tags = numTags
        self.title_out_weight = tf.get_variable(
            "title_out_weight",
            shape=[self.embedding_size*2, self.num_tags],
            initializer=modeling.create_initializer(self.bert_config.initializer_range))
        self.title_out_bias = tf.get_variable(
            "title_out_bias", shape=[self.num_tags])
        self.type_embedding = tf.get_variable(
            "ner_types",
            shape=[3, self.embedding_size],
            initializer=modeling.create_initializer(self.bert_config.initializer_range))
        

    def length(self, data, axis=1):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=axis)
        length = tf.cast(length, tf.int32)
        return length

    def biLSTMEncode(self, X, xLen):
        with tf.variable_scope("biLSTM") as scope:
            bcell = tf.nn.rnn_cell.GRUCell(
                num_units=self.embedding_size)
            fcell = tf.nn.rnn_cell.GRUCell(
                num_units=self.embedding_size)
            bcell = tf.nn.rnn_cell.DropoutWrapper(
                cell=bcell, output_keep_prob=1.0-self.dropout_h)
            fcell = tf.nn.rnn_cell.DropoutWrapper(
                cell=fcell, output_keep_prob=1.0-self.dropout_h)
            outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                bcell,
                fcell,
                X,
                sequence_length=xLen,
                dtype=tf.float32,
                time_major=False,
                scope="biLSTM")
            return tf.concat(outputs,axis=2)

    def inference(self, name=None):
        with tf.variable_scope("tf_inference"):
            X = self.sentence_placeholder
            shapeS = tf.shape(X)
            padCLS = tf.ones([shapeS[0], 1], dtype=tf.int32) * (kVocabSize-1)
            paddedX = tf.concat([padCLS, X], axis=1)

            amask = tf.sequence_mask(
                self.a_length, self.max_token_per_sentence+1, dtype=tf.int32)
            self.totalLength = self.a_length
            abmask = tf.sequence_mask(
                self.totalLength, self.max_token_per_sentence+1, dtype=tf.int32)
            totalmask = amask + abmask
            self.bert = BertModel(config=self.bert_config, input_ids=paddedX, input_mask=abmask,
                                  token_type_ids=totalmask, use_one_hot_embeddings=False, droprate=self.dropout_h)
        X = self.bert.get_sequence_output()
        X = X[:, 1:, :]
        types = tf.nn.embedding_lookup(self.type_embedding, self.types_placeholder)

        # X = X + types

        # X = self.biLSTMEncode(X, self.totalLength-1)
        output = tf.reshape(X, shape=[-1, self.embedding_size])
        # output = tf.stop_gradient(output)
        with tf.variable_scope("tag/cls"):
            output = tf.layers.dense(
                output,
                units=self.embedding_size*2,
                activation=modeling.get_activation(
                    self.bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    self.bert_config.initializer_range))
            output = modeling.layer_norm(output)
        # output = tf.nn.dropout(output, 1 - self.dropout_h)
        output = tf.nn.xw_plus_b(
            output, self.title_out_weight, self.title_out_bias)
        unary_scores = tf.reshape(
            output, [-1, self.max_token_per_sentence, self.num_tags],
            name="finalInference")
        return unary_scores, self.totalLength-1

    def loss(self, logits, pLen):

        # mask = tf.sequence_mask(pLen, self.max_token_per_sentence,dtype=tf.float32)

        # loss = tf.contrib.seq2seq.sequence_loss(logits, self.targets, self.weights_placeholder)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, tf.cast(self.targets, tf.int32), pLen)
        loss = tf.reduce_mean(-log_likelihood)
        # self.transition_params = tf.ones([self.num_tags, self.num_tags], name="transitions")
        return loss
