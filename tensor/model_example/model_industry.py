# -*- coding: utf-8 -*-
# @Date:   2019-06-04 16:04:56
# @Last Modified time: 2019-07-18 09:59:23
import tensorflow as tf
import math
import numpy as np
from tensor.transformer.bert.bert_official import BertConfig, BertModel

kCheckMaxLen = 128
kVocabSize = 32003

class Model(object):
    def __init__(self,
                 tag_num,
                 num_hidden=1024):
        self.target = tf.sparse.placeholder(tf.int64, shape=[None,], name="target")
        self.targets = tf.placeholder(tf.float32, shape=[None, tag_num], name="targets")
        self.sentence_holder = tf.placeholder(
            tf.int32, shape=[None, kCheckMaxLen], name="sentence")
        self.check_alen_holder = tf.placeholder(

            tf.int32, shape=[None,], name="check_alen")
        self.check_tlen_holder = tf.placeholder(
            tf.int32, shape=[None,], name="check_tlen")
        self.hidden_weights = tf.get_variable(
            "hidden_weight",
            shape=[200, num_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        self.hidden_bias = tf.get_variable("hidden_bias", shape=[num_hidden])
        self.out_weight = tf.get_variable(
            "out_weight",
            shape=[num_hidden, tag_num],
            initializer=tf.contrib.layers.xavier_initializer())
        self.out_bias = tf.get_variable(
            "out_bias", shape=[tag_num], initializer=tf.constant_initializer([0.05,-0.05]))
        self.dropout_h = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.bert_config = BertConfig(
            kVocabSize, hidden_size=200, num_hidden_layers=6, num_attention_heads=8, intermediate_size=800)

        self.preds = self.inference()
        self.preds = tf.identity(self.preds, name='final_inference')

    def gelu(self, input_tensor):
        """Gaussian Error Linear Unit.
        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
            input_tensor: float Tensor to perform activation.
        Returns:
            `input_tensor` with the GELU activation applied.
        """
        # cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
        # return input_tensor * cdf
        return tf.nn.elu(input_tensor)

    def inference(self, name=None):
        with tf.variable_scope("tf_inference"):
            line = tf.reshape(self.sentence_holder, [-1, kCheckMaxLen])
            shapeS = tf.shape(line)
            padCLS = tf.ones([shapeS[0], 1], dtype=tf.int32) * (kVocabSize-1)
            paddedLines = tf.concat([padCLS, line], axis=1)
            a_length = tf.reshape(self.check_alen_holder, [-1])
            a_length = a_length + 1
            amask = tf.sequence_mask(
                a_length, kCheckMaxLen+1, dtype=tf.int32)
            b_length = tf.reshape(self.check_tlen_holder, [-1])
            b_length = b_length + 1
            bmask = tf.sequence_mask(
                b_length, kCheckMaxLen+1, dtype=tf.int32)
            totalmask = amask + bmask
            self.bertLine = BertModel(config=self.bert_config, input_ids=paddedLines, input_mask=bmask,
                                      token_type_ids=totalmask, use_one_hot_embeddings=False, droprate=self.dropout_h, scope='bert')
        pooled_out = self.bertLine.get_pooled_output()

        finalHidden = tf.nn.xw_plus_b(
            pooled_out, self.hidden_weights, self.hidden_bias)
        finalHidden = self.gelu(finalHidden)
        finalHidden = tf.nn.dropout(finalHidden, 1.0-self.dropout_h)
        pred = tf.nn.xw_plus_b(
            finalHidden, self.out_weight, self.out_bias)
        return pred
