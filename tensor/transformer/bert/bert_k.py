#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: bert_k.py

import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
import numpy as np
import math
import optimization
from transformer import model as TransformerModel, shape_list, positional_encoding


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
        "cls_target":
        tf.FixedLenFeature([], tf.int64, default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    sentence = parsed_features["sentence"]
    sentence.set_shape([199])
    label_targets = parsed_features["label_target"]
    label_targets.set_shape([26])
    label_indexies = parsed_features["label_index"]
    label_indexies.set_shape([26])
    length = parsed_features["length"]
    cls_target = parsed_features["cls_target"]
    return sentence, label_targets, label_indexies, length, cls_target


class Model(object):
    def __init__(self,
                 maxTokenPerSetence,
                 wordsEm,
                 embeddingSize,
                 vocabSize,
                 lastHiddenSize=200,
                 positionEmbeddingSize=20):
        self.max_token_per_sentence = 199
        self.max_labels = 26
        self.pos_embedding_size = positionEmbeddingSize
        self.sentence_placeholder = tf.placeholder(
            tf.int32,
            shape=[None, self.max_token_per_sentence],
            name="sentence")
        self.label_target = tf.placeholder(
            tf.int32, shape=[None, self.max_labels])
        self.label_index = tf.placeholder(
            tf.int32, shape=[None, self.max_labels])
        self.cls_target = tf.placeholder(tf.int32, shape=[None])
        self.totalLength = tf.placeholder(tf.int32, shape=[None])
        self.words = tf.Variable(wordsEm, name="words")
        self.embedding_size = embeddingSize + \
            self.pos_embedding_size
        self.cls = tf.get_variable(
            "CLS", [embeddingSize], initializer=tf.truncated_normal_initializer())
        self.last_hidden_size = lastHiddenSize
        self.learning_rate_h = tf.placeholder(tf.float32, shape=[])
        self.dropout_h = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.sample_size_for_lm = 12000
        self.vocab_size = vocabSize
        with tf.variable_scope("lm") as scope:
            self.languge_model_weights = tf.get_variable(
                "languge_model_weights",
                shape=[self.embedding_size, self.vocab_size],
                # regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.contrib.layers.xavier_initializer())
            self.languge_model_bias = tf.get_variable(
                "languge_model_bias", shape=[self.vocab_size])

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, name=None):
        with tf.variable_scope("tf_inference"):
            X = self.sentence_placeholder
            seqEmds = tf.nn.embedding_lookup(self.words, X)
            shapeS = tf.shape(seqEmds)
            clsH = tf.tile(self.cls, [shapeS[0]])
            clsH = tf.reshape(clsH, [
                              shapeS[0], self.embedding_size-self.pos_embedding_size])
            clsH = tf.expand_dims(clsH, axis=1)
            X = tf.concat([clsH, seqEmds], axis=1)
            xs = tf.tile([0], [shapeS[0]])
            xs = tf.reshape(xs, [shapeS[0], 1])
            Xpos = positional_encoding(
                tf.concat([xs, self.sentence_placeholder], axis=1), self.pos_embedding_size)
            totalmask = tf.sequence_mask(
                self.totalLength, self.max_token_per_sentence+1, dtype=tf.int32)
            X = tf.concat([X, Xpos], axis=2)
            print("now X is: %r" % (X))
            X = TransformerModel(10, 3, X, self.dropout_h, mask=totalmask)
        return X

    def train(self, loss, lr, totalSteps, warnSteps):

        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_h)
        # gradients, variables = zip(*optimizer.compute_gradients(loss))
        # gradients = [
        #     None if gradient is None else tf.clip_by_norm(gradient, 5.0)
        #     for gradient in gradients
        # ]
        # train_op = optimizer.apply_gradients(zip(gradients, variables))
        # return train_op
        return optimization.create_optimizer(
            loss, lr, totalSteps, warnSteps, False)

    def loss(self):
        X = self.inference(name="inference_final")
        labelLen = self.length(self.label_index)
        shapeS = tf.shape(self.label_index)
        zeroLabels = tf.tile([0], [shapeS[0]])
        zeroLabels = tf.reshape(zeroLabels, [shapeS[0], 1])
        appendLabelIndex = tf.concat([zeroLabels, self.label_index], axis=1)
        todoX = batch_gather(X, appendLabelIndex)
        appendLabelTarget = tf.concat(
            [tf.reshape(self.cls_target, [-1, 1]), self.label_target], axis=1)
        logits = tf.nn.xw_plus_b(tf.reshape(
            todoX, [-1, self.embedding_size]), self.languge_model_weights, self.languge_model_bias)
        mlmCost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(appendLabelTarget, [-1]),
            logits=logits
        )
        preds = tf.reshape(tf.arg_max(logits, dimension=1,
                                      output_type=tf.int32), [-1, self.max_labels+1])

        sameCounts = tf.to_float(tf.equal(preds, appendLabelTarget))

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
            tf.sequence_mask(labelLen+1, self.max_labels+1), tf.float32)
        sameCounts = sameCounts * lengthMask
        sameCounts = tf.reduce_sum(sameCounts)
        totalCount = tf.to_float(tf.reduce_sum(labelLen + 1))
        accuracy = sameCounts / totalCount
        # print("lengthMask: %r" % (lengthMask))
        mlmCost = tf.reshape(mlmCost, [-1, self.max_labels+1])
        mlmCost = mlmCost * lengthMask

        mlmCost = tf.reduce_sum(mlmCost, axis=1, keepdims=True)
        mlmCost = mlmCost / tf.add(tf.cast(labelLen, tf.float32), 1.0)
        mlmCost = tf.reduce_mean(mlmCost)

        return mlmCost, sameCounts, accuracy
