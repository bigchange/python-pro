#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: bert_ner.py
# Project: bert

# Copyright 2020 - 2018



import tensorflow as tf
import numpy as np
import math
from transformer import model as TransformerModel, shape_list, positional_encoding

kMaxSeqLen = 200
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
        "length":
        tf.FixedLenFeature([], tf.int64, default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    sentence = parsed_features["sentence"]
    sentence.set_shape([kMaxSeqLen-1])
    targets = parsed_features["targets"]
    targets.set_shape([kMaxSeqLen-1])
    length = parsed_features["length"]+1
    return sentence, targets, length

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = tf.nn.dropout(output_tensor, 1-dropout_prob)
  return output_tensor

class Model(object):
    def __init__(self,
                 embeddingSize,
                 numTags,
                 lastHiddenSize=200,
                 positionEmbeddingSize=18,
                 segEmbeddingSize=2):
        self.max_token_per_sentence = kMaxSeqLen-1
        self.pos_embedding_size = positionEmbeddingSize
        self.seg_embedding_size = segEmbeddingSize
        self.sentence_placeholder = tf.placeholder(
            tf.int32,
            shape=[None, self.max_token_per_sentence],
            name="sentence")
        self.targets = tf.placeholder(
            tf.int32, shape=[None, self.max_token_per_sentence])
        self.a_length = tf.placeholder(tf.int32, shape=[None],name="alength")
        self.totalLength = self.a_length
        self.words = tf.get_variable(
            "words", [kVocabSize, embeddingSize], initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.embedding_size = embeddingSize + \
            self.pos_embedding_size + self.seg_embedding_size
        self.cls = tf.get_variable(
            "CLS", [embeddingSize], initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.segs = tf.get_variable(
            "ABN", [3, self.seg_embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.last_hidden_size = lastHiddenSize
        self.learning_rate_h = tf.placeholder(tf.float32, shape=[])
        self.dropout_h = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.num_tags = numTags
        self.title_out_weight = tf.get_variable(
            "title_out_weight",
            shape=[self.embedding_size, self.num_tags],
            initializer=tf.contrib.layers.xavier_initializer())
        self.title_out_bias = tf.get_variable(
            "title_out_bias", shape=[self.num_tags])

    def length(self, data, axis=1):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=axis)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, name=None):
        with tf.variable_scope("tf_inference"):
            X = self.sentence_placeholder
            seqEmds = tf.nn.embedding_lookup(self.words, X)
            shapeS = tf.shape(seqEmds)

            clsH = tf.tile(self.cls, [shapeS[0]])
            clsH = tf.reshape(clsH, [
                              shapeS[0], self.embedding_size-self.pos_embedding_size-self.seg_embedding_size])
            clsH = tf.expand_dims(clsH, axis=1)


            # now X is [Batch, kMaxSeqLen, kMaxSubToken, embedding]
            X = tf.concat([clsH, seqEmds], axis=1)
            xs = tf.zeros([shapeS[0], kMaxSeqLen])
            #[Batch, kMaxSeqLen, embedding2]
            Xpos = positional_encoding(xs, self.pos_embedding_size)
            
            amask = tf.sequence_mask(
                self.a_length, self.max_token_per_sentence+1, dtype=tf.int32)
            abmask = tf.sequence_mask(
                self.totalLength, self.max_token_per_sentence+1, dtype=tf.int32)
            totalmask = amask + abmask
            segE = tf.nn.embedding_lookup(self.segs, totalmask)
           

            X = tf.concat([X, Xpos, segE], axis=2)
            X = layer_norm_and_dropout(X, self.dropout_h)
            print("now X is: %r" % (X))
            X = TransformerModel(8, 6, X, self.dropout_h, mask=abmask)
        X = X[:,1:,:]
        output = tf.reshape(X, shape=[-1, self.embedding_size])
        output = tf.nn.xw_plus_b(
                output, self.title_out_weight, self.title_out_bias)
        unary_scores = tf.reshape(
                output, [-1, self.max_token_per_sentence, self.num_tags],
                name="finalInference")
        return unary_scores, self.totalLength-1

    def loss(self,logits,pLen):
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            logits , tf.cast(self.targets, tf.int32), pLen)
        loss = tf.reduce_mean(-log_likelihood)
        return loss
