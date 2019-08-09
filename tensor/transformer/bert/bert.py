#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
import numpy as np
import math
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
        "a_length":
        tf.FixedLenFeature([], tf.int64, default_value=0),
        "ab_length":
        tf.FixedLenFeature([], tf.int64, default_value=0),
        "nsp_target":
        tf.FixedLenFeature([], tf.int64, default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    sentence = parsed_features["sentence"]
    sentence.set_shape([199])
    label_targets = parsed_features["label_target"]
    label_targets.set_shape([30])
    label_indexies = parsed_features["label_index"]
    label_indexies.set_shape([30])
    a_length = parsed_features["a_length"]
    ab_length = parsed_features["ab_length"]
    nsp_target = parsed_features["nsp_target"]
    return sentence, label_targets, label_indexies, a_length, ab_length, nsp_target


class Model(object):
    def __init__(self,
                 maxTokenPerSetence,
                 wordsEm,
                 embeddingSize,
                 vocabSize,
                 lastHiddenSize=200,
                 positionEmbeddingSize=18,
                 segEmbeddingSize=2):
        self.max_token_per_sentence = 199
        self.max_labels = 30
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
        self.nsp_target = tf.placeholder(tf.int32, shape=[None])
        self.a_length = tf.placeholder(tf.int32, shape=[None])
        self.ab_length = tf.placeholder(tf.int32, shape=[None])
        self.words = tf.Variable(wordsEm, name="words")
        self.embedding_size = embeddingSize + \
            self.pos_embedding_size + self.seg_embedding_size
        self.cls = tf.get_variable(
            "CLS", [embeddingSize], initializer=tf.truncated_normal_initializer())
        self.segs = tf.get_variable(
            "ABN", [3, self.seg_embedding_size], initializer=tf.truncated_normal_initializer())
        self.last_hidden_size = lastHiddenSize
        self.learning_rate_h = tf.placeholder(tf.float32, shape=[])
        self.dropout_h = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.sample_size_for_lm = 15000
        self.vocab_size = vocabSize
        with tf.variable_scope("lm") as scope:
            self.languge_model_weights = tf.get_variable(
                "languge_model_weights",
                shape=[self.vocab_size, self.embedding_size],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.contrib.layers.xavier_initializer())
            self.languge_model_bias = tf.get_variable(
                "languge_model_bias", shape=[self.vocab_size])
        with tf.variable_scope("nsp") as scope:
            self.nsp_weights = tf.get_variable(
                "nsp_weights",
                shape=[self.embedding_size, 2],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.contrib.layers.xavier_initializer())
            self.nsp_bias = tf.get_variable(
                "nsp_bias", shape=[2])

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
            # self.embedding_size-self.pos_embedding_size-self.seg_embedding_size] == embeddingSize
            clsH = tf.reshape(clsH, [
                              shapeS[0], self.embedding_size-self.pos_embedding_size-self.seg_embedding_size])
            clsH = tf.expand_dims(clsH, axis=1)
            # add [cls]的embedding
            X = tf.concat([clsH, seqEmds], axis=1)
            # position add [cls]: cls + max_token_per_sentence = final_length = 200
            xs = tf.tile([0], [shapeS[0]])
            xs = tf.reshape(xs, [shapeS[0], 1])
            # get position的embedding
            Xpos = positional_encoding(
                tf.concat([xs, self.sentence_placeholder], axis=1), self.pos_embedding_size)
            amask = tf.sequence_mask(
                self.a_length, self.max_token_per_sentence+1, dtype=tf.int32)
            abmask = tf.sequence_mask(
                self.ab_length, self.max_token_per_sentence+1, dtype=tf.int32)
            totalmask = amask + abmask
            # add segment id 的embedding, default 0, if is a sentence then 2, else if is b sentence then 1
            segE = tf.nn.embedding_lookup(self.segs, totalmask)
            X = tf.concat([X, Xpos, segE], axis=2)
            print("now X is: %r" % (X))
            # 10 heads， 2 layers,  abmask: 记录输入的有效token位置
            X = TransformerModel(10, 3, X, self.dropout_h, mask=abmask)

        return X

    def train(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_h)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, 5.0)
            for gradient in gradients
        ]
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op

    def loss(self):
        X = self.inference(name="inference_final")
        labelLen = self.length(self.label_index)
        todoX = batch_gather(X, self.label_index)

        mlmCost = tf.nn.sampled_softmax_loss(
            self.languge_model_weights,
            self.languge_model_bias,
            tf.reshape(self.label_target, [-1, 1]),
            tf.reshape(todoX, [-1, self.embedding_size]),
            self.sample_size_for_lm,
            self.vocab_size,
            partition_strategy='div',
        )
        # print("mlmCost: %r" % (mlmCost))
        lengthMask = tf.cast(
            tf.sequence_mask(labelLen, self.max_labels), tf.float32)
        # print("lengthMask: %r" % (lengthMask))
        mlmCost = tf.reshape(mlmCost, [-1, self.max_labels])
        mlmCost = mlmCost * lengthMask
        mlmCost = tf.reduce_sum(mlmCost, axis=1, keepdims=True)
        mlmCost = mlmCost / tf.add(tf.cast(labelLen, tf.float32),0.00001)
        mlmCost = tf.reduce_mean(mlmCost)
        clsHidden = tf.slice(X, [0, 0, 0], [-1, 1, -1])
        clsHidden = tf.reshape(clsHidden, [-1, self.embedding_size])
        preds = tf.nn.xw_plus_b(clsHidden, self.nsp_weights, self.nsp_bias)
        predLabels = tf.argmax(preds, axis=1)
        correct = tf.reduce_sum(
            tf.cast(tf.equal(tf.cast(predLabels, tf.int32), self.nsp_target), tf.int32))

        nspCost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.nsp_target, logits=preds))
        cost = nspCost + mlmCost
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cost += tf.reduce_sum(reg_losses)
        return cost, correct
