#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: multiturn_model.py
# Project: transformer
# Copyright 2020 - 2019


import numpy as np
import tensorflow as tf
import optimization
from transformer import Transformer, LayerNormalization
from collections import defaultdict
import model_utils


BASE_PARAMS = defaultdict(
    lambda: None,  # Set default value to None.

    # Input params
    default_batch_size=512,  # Maximum number of tokens per batch of examples.
    default_batch_size_tpu=32768,
    max_length=300,  # Maximum number of tokens per example.

    # Model params
    initializer_gain=2.0,  # Used in trainable variable initialization.
    initializer_range = 0.02,
    vocab_size=32003,  # Number of tokens defined in the vocabulary file.
    hidden_size=200,  # Model dimension in the hidden layers.
    num_hidden_layers=6,  # Number of layers in the encoder and decoder stacks.
    num_heads=8,  # Number of heads to use in multi-headed attention.
    filter_size=800,  # Inner layer dimension in the feedforward network.

    # Dropout values (only used when training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    # Training params
    label_smoothing=0.1,
    learning_rate=2.0,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=16000,

    # Optimizer params
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

    # Default prediction params
    extra_decode_length=90,
    beam_size=4,
    alpha=0.6,  # used to calculate length normalization initializer_rangen beam search
    
    num_types = 3, # how many id types
    # TPU specific parameters
    use_tpu=False,
    static_batch=False,
    allow_ffn_pad=True,
)

class MultiTurnModel:
    def __init__(self, maxLength):
        self.transformer_conf=BASE_PARAMS
        self.vocab_size = self.transformer_conf['vocab_size']
        self.embed_dim = self.transformer_conf['hidden_size']
        self.num_units = self.transformer_conf['hidden_size']
        self.max_length = maxLength
        self.y = tf.placeholder(
            tf.int32, shape=[None], name="target")
        self.inp_X = tf.placeholder(
            tf.int32, shape=[None, self.max_length], name="inp_x")
        self.partTypes = tf.placeholder(
            tf.int32, shape=[None, self.max_length], name="part_types")
        self.a_length = tf.placeholder(tf.int32, shape=[None], name="alength")
        self.totalLength = tf.placeholder(
            tf.int32, shape=[None], name="totalLength")
        self.out_weight = tf.get_variable(
            "out_weight",
            shape=[self.num_units, 1],
            initializer=tf.contrib.layers.xavier_initializer())
        self.out_bias = tf.get_variable(
            "out_bias", shape=[1])


      
    def inference(self,transformer, name=None, training=True):
        with tf.variable_scope("tf_inference"):
            X = self.inp_X
            shapeS = tf.shape(X)
            padCLS = tf.ones([shapeS[0], 1], dtype=tf.int32) * (self.vocab_size-1)
            padParts = tf.zeros([shapeS[0], 1], dtype=tf.int32)
            
            paddedX = tf.concat([padCLS, X], axis=1)
            paddedParts = tf.concat([padParts, self.partTypes], axis=1)
            partAttBias = model_utils.get_utterance_attention_bias(paddedParts)
            amask = tf.sequence_mask(
                self.a_length+1, self.max_length+1, dtype=tf.int32)
            abmask = tf.sequence_mask(
                self.totalLength+1, self.max_length+1, dtype=tf.int32)
            totalmask = amask + abmask
            _, results = transformer(paddedX,totalmask,encodeAttMask=partAttBias)
            
            aR = tf.expand_dims(tf.to_float(amask), axis=-1)*results
            aR = tf.math.reduce_sum(aR,axis=1)
            aR = aR / tf.expand_dims(tf.to_float(self.a_length+1), axis=-1)
            
            bmask = abmask - amask
            bR = tf.expand_dims(tf.to_float(bmask), axis=-1)*results
            bR = tf.math.reduce_sum(bR,axis=1)
            bR = bR / tf.expand_dims(tf.to_float(self.totalLength-self.a_length), axis=-1)
            
          
            pool = results[:,0,:]
            
            ret = tf.nn.xw_plus_b(pool, self.out_weight, self.out_bias)
            ret = tf.squeeze(ret, axis=1, name=name)
            
        return ret

    def loss(self,training=True):
        transformer = Transformer(self.transformer_conf, training)
        preds = self.inference(transformer,name="final_inference", training=training)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.y), logits=preds)
        loss = tf.reduce_mean(loss)
        
        predsY = tf.to_int32(tf.greater(preds,0))
        
        sameCount= tf.to_int32(tf.equal(predsY, self.y))
        sameCount = tf.reduce_sum(sameCount)
        
      
        return loss, sameCount, preds