#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: model_use_official.py
# Project: bert

# Copyright 2020 - 2019


import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
import numpy as np
import math
import optimization
import bert_official as modeling
from bert_official import BertConfig, BertModel
kMaxSeqLen = 200
kMaxSubToken = 3
kMaxTarget = 3
kMaxLabel = 26
kVocabSize = 32003


def batch_gather(params, indices, name=None):
    """Gather slices from `params` according to `indices` with leading batch dims.
    This operation assumes that the leading dimensions of `indices` are dense,
    and the gathers on the axis corresponding to the last dimension of `indices`.
    More concretely it computes:
    result[i1, ..., in] = params[i1, ..., in-1, indices[i1, ..., in]]
    Therefore `params` should be a Tensor of shape [A1, ..., AN, B1, ..., BM],
    `indices` should be a Tensor of shape [A1, ..., AN-1, C] and `result` will be
    a Tensor of size `[A1, ..., AN-1, C, B1, ..., BM]`.
    In the case in which indices is a 1D tensor, this operation is equivalent to
    `tf.gather`.
    See also `tf.gather` and `tf.gather_nd`.
    Args:
      params: A Tensor. The tensor from which to gather values.
      indices: A Tensor. Must be one of the following types: int32, int64. Index
          tensor. Must be in range `[0, params.shape[axis]`, where `axis` is the
          last dimension of `indices` itself.
      name: A name for the operation (optional).
    Returns:
      A Tensor. Has the same type as `params`.
    Raises:
      ValueError: if `indices` has an unknown shape.
    """

    with tf.name_scope(name):
        indices = tf.convert_to_tensor(indices, name="indices")
        params = tf.convert_to_tensor(params, name="params")
        indices_shape = tf.shape(indices)
        params_shape = tf.shape(params)
        ndims = indices.shape.ndims
        if ndims is None:
            raise ValueError("batch_gather does not allow indices with unknown "
                             "shape.")
        batch_indices = indices
        accum_dim_value = 1
        for dim in range(ndims-1, 0, -1):
            dim_value = params_shape[dim-1]
            accum_dim_value *= params_shape[dim]
            dim_indices = tf.range(0, dim_value, 1)
            dim_indices *= accum_dim_value
            dim_shape = tf.stack([1] * (dim - 1) + [dim_value] + [1] * (ndims - dim),
                                 axis=0)
            batch_indices += tf.reshape(dim_indices, dim_shape)

        flat_indices = tf.reshape(batch_indices, [-1])
        outer_shape = params_shape[ndims:]
        flat_inner_shape = gen_math_ops.prod(
            params_shape[:ndims], [0], False)

        flat_params = tf.reshape(
            params, tf.concat([[flat_inner_shape], outer_shape], axis=0))
        flat_result = tf.gather(flat_params, flat_indices)
        result = tf.reshape(flat_result, tf.concat(
            [indices_shape, outer_shape], axis=0))
        final_shape = indices.get_shape()[:ndims-1].merge_with(
            params.get_shape()[:ndims - 1])
        final_shape = final_shape.concatenate(indices.get_shape()[ndims-1])
        final_shape = final_shape.concatenate(params.get_shape()[ndims:])
        result.set_shape(final_shape)
        return result


def parse_tfrecord_function(example_proto):
    features = {
        "label_target":
        tf.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True, default_value=0),
        "label_index":
        tf.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True, default_value=0),
        "sentence":
        tf.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True, default_value=0),
        "length":
        tf.FixedLenFeature([], tf.int64, default_value=0),
        "alength":
        tf.FixedLenFeature(
            [], tf.int64, default_value=0),
        "target":
        tf.FixedLenFeature(
            [], tf.int64, default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    sentence = parsed_features["sentence"]
    sentence.set_shape([(kMaxSeqLen-1)])
    sentence = tf.reshape(sentence, [kMaxSeqLen-1])
    label_targets = parsed_features["label_target"]
    label_targets.set_shape([kMaxLabel])
    label_indexies = parsed_features["label_index"]
    label_indexies.set_shape([kMaxLabel])
    length = parsed_features["length"]+1
    alength = parsed_features["alength"]+1
    target = parsed_features["target"]

    return sentence, label_targets, label_indexies, length, alength, target


class Model(object):
    def __init__(self,
                 embeddingSize,
                 vocabSize=kVocabSize,
                 lastHiddenSize=200,
                 positionEmbeddingSize=18,
                 segEmbeddingSize=2):
        self.max_token_per_sentence = kMaxSeqLen-1
        self.max_labels = kMaxLabel
        self.pos_embedding_size = positionEmbeddingSize
        self.seg_embedding_size = segEmbeddingSize
        self.sentence_placeholder = tf.placeholder(
            tf.int32,
            shape=[None, self.max_token_per_sentence],
            name="sentence")
        self.label_target = tf.placeholder(
            tf.int32, shape=[None, self.max_labels])
        self.label_index = tf.placeholder(
            tf.int32, shape=[None, self.max_labels])
        self.target = tf.placeholder(tf.int32, shape=[None])
        self.totalLength = tf.placeholder(
            tf.int32, shape=[None], name="totalLength")
        self.a_length = tf.placeholder(tf.int32, shape=[None], name="alength")
        self.embedding_size = embeddingSize + \
            self.pos_embedding_size + self.seg_embedding_size
        self.bert_config = BertConfig(
            kVocabSize, hidden_size=200, num_hidden_layers=6, num_attention_heads=8, intermediate_size=800)
        self.last_hidden_size = lastHiddenSize
        self.learning_rate_h = tf.placeholder(tf.float32, shape=[])
        self.dropout_h = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.vocab_size = vocabSize
        with tf.variable_scope("lm") as scope:
            # self.languge_model_weights = tf.get_variable(
            #     "w_languge_model_weights",
            #     shape=[self.embedding_size, self.vocab_size],
            #     # regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            #     initializer=tf.contrib.layers.xavier_initializer())
            self.languge_model_bias = tf.get_variable(
                "w_languge_model_bias", shape=[self.vocab_size])
            self.final_weights = tf.get_variable(
                "final_weights",
                shape=[self.embedding_size, 2],
                # regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.contrib.layers.xavier_initializer())
            self.final_bias = tf.get_variable(
                "final_bias", shape=[2])

    def length(self, data, axis=1):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=axis)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, name=None, training=True):
        with tf.variable_scope("tf_inference"):
            X = self.sentence_placeholder
            shapeS = tf.shape(X)
            padCLS = tf.ones([shapeS[0], 1], dtype=tf.int32) * (kVocabSize-1)
            paddedX = tf.concat([padCLS, X], axis=1)

            amask = tf.sequence_mask(
                self.a_length, self.max_token_per_sentence+1, dtype=tf.int32)
            abmask = tf.sequence_mask(
                self.totalLength, self.max_token_per_sentence+1, dtype=tf.int32)
            totalmask = amask + abmask
            self.bert = BertModel(config=self.bert_config, input_ids=paddedX, input_mask=abmask,
                                  token_type_ids=totalmask, use_one_hot_embeddings=False, droprate=self.dropout_h, scope="bert")
        return self.bert.get_sequence_output(), tf.identity(self.bert.get_pooled_output(),name="repr")

    def train(self, loss, lr, totalSteps, warnSteps):
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_h)
        # gradients, variables = zip(*optimizer.compute_gradients(loss))
        # gradients = [
        #     None if gradient is None else tf.clip_by_norm(gradient, 5.0)
        #     for gradient in gradientss
        # ]
        # train_op = optimizer.apply_gradients(zip(gradients, variables))
        # return train_op
        return optimization.create_optimizer(
            loss, lr, totalSteps, warnSteps, False)

    def loss(self):
        X, pooled = self.inference(name="inference_final")
        labelLen = self.length(self.label_index)
        shapeS = tf.shape(self.label_index)

        todoX = batch_gather(X, self.label_index)

        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            todoX = tf.layers.dense(
                todoX,
                units=self.embedding_size,
                activation=modeling.get_activation(
                    self.bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    self.bert_config.initializer_range))
        todoX = modeling.layer_norm(todoX)
        embeddings = self.bert.get_embedding_table()
        logits = tf.nn.xw_plus_b(tf.reshape(
            todoX, [-1, self.embedding_size]), tf.transpose(embeddings), self.languge_model_bias)

        mlmCost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.label_target, [-1]),
            logits=logits
        )
        preds = tf.reshape(tf.arg_max(logits, dimension=1,
                                      output_type=tf.int32), [-1, self.max_labels])

        sameCounts = tf.to_float(tf.equal(preds, self.label_target))

        # mlmCost = tf.nn.sampled_softmax_loss(
        #     self.languge_model_weights,
        #     self.languge_model_bias,
        #     tf.reshape(appendLabelTarget, [-1, 1]),
        #     tf.reshape(todoX, [-1, self.embedding_size]),
        #     self.sample_size_for_lm,
        #     self.vocab_size,
        #     partition_strategy='div',
        # )
        # print("mlmCost: %r" % (mlmCost))
        lengthMask = tf.cast(
            tf.sequence_mask(labelLen, self.max_labels), tf.float32)
        sameCounts = sameCounts * lengthMask
        sameCounts = tf.reduce_sum(sameCounts)
        totalCount = tf.to_float(tf.reduce_sum(labelLen))
        accuracy = sameCounts / totalCount
        # print("lengthMask: %r" % (lengthMask))
        mlmCost = tf.reshape(mlmCost, [-1, self.max_labels])
        mlmCost = mlmCost * lengthMask

        mlmCost = tf.reduce_sum(mlmCost, axis=1)
        mlmCost = mlmCost / tf.cast(labelLen, tf.float32)
        mlmCost = tf.reduce_mean(mlmCost)

        predH = tf.nn.xw_plus_b(
            pooled, self.final_weights, self.final_bias)
        headCost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predH,
            labels=self.target
        )
        headCost = tf.reduce_mean(headCost)
        predH = tf.arg_max(predH, dimension=1,
                           output_type=tf.int32)
        accuracy2 = tf.to_float(tf.equal(predH, self.target))
        accuracy2 = tf.reduce_sum(accuracy2) / tf.to_float(shapeS[0])
        return mlmCost+headCost, sameCounts, accuracy, accuracy2
