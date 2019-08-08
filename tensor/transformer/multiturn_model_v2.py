#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: multiturn_model_v2.py
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
    layer_postprocess_dropout=0.2,
    attention_dropout=0.2,
    relu_dropout=0.2,

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

kMaxCheck = 9

class MultiTurnModelV2:
    def __init__(self, maxLength, numAspect=200):
        self.transformer_conf=BASE_PARAMS
        self.vocab_size = self.transformer_conf['vocab_size']
        self.embed_dim = self.transformer_conf['hidden_size']
        self.num_units = self.transformer_conf['hidden_size']
        self.max_length = maxLength
        self.y = tf.placeholder(
            tf.int32, shape=[None], name="target")
        self.inp_X = tf.placeholder(
            tf.int32, shape=[None,  kMaxCheck, self.max_length], name="inp_x")
        self.a_length = tf.placeholder(tf.int32, shape=[None, kMaxCheck], name="alength")
        self.t_length = tf.placeholder(tf.int32, shape=[None, kMaxCheck], name="tlength")
        self.seg_types = tf.placeholder(tf.int32, shape=[None, kMaxCheck], name="seg_types")
        self.ln = LayerNormalization(self.num_units)
        self.out_weight = tf.get_variable(
            "out_weight",
            shape=[self.num_units*2, 1],
            initializer=tf.contrib.layers.xavier_initializer())
        self.out_bias = tf.get_variable(
            "out_bias", shape=[1])
        self.num_aspect = numAspect
        self.match_proj_weight = tf.get_variable(
            "match_proj_weight",
            shape=[self.num_units, self.num_aspect],
            initializer=tf.contrib.layers.xavier_initializer())
        self.match_proj_bias = tf.get_variable(
            "match_proj_bias", shape=[self.num_aspect])


      
    def inference(self,transformer, name=None, training=True):
        with tf.variable_scope("tf_inference", reuse=tf.AUTO_REUSE):
            X = self.inp_X
            shapeS = tf.shape(X)
            bsz = shapeS[0]
            X = tf.reshape(X, [-1, self.max_length])
            alens = tf.reshape(self.a_length, [-1])
            tlens = tf.reshape(self.t_length, [-1])
            
            padCLS = tf.ones([bsz*kMaxCheck, 1], dtype=tf.int32) * (self.vocab_size-1)
            paddedX = tf.concat([padCLS, X], axis=1)
            
            amask = tf.sequence_mask(
                alens+1, self.max_length+1, dtype=tf.int32)
            abmask = tf.sequence_mask(
                tlens+1, self.max_length+1, dtype=tf.int32)
            
            totalmask = amask + abmask
            _, results = transformer(paddedX,totalmask)
            
            results = tf.reshape(results,[bsz, kMaxCheck, self.max_length+1, self.num_units])
            #[bsz, maxcheck, hiddens]
            results = results[:,:,0,:]
            
            seglen = tf.to_int32(tf.greater(self.seg_types,0))
            seglen = tf.reduce_sum(seglen, axis=1)
            
            #[bsz, maxcheck, hiddens]
            segTypes = model_utils.get_input_types(self.seg_types, self.num_units,num_types=8,name="qa_types")
            
            
            results =self.ln(results+segTypes)
            results = tf.reshape(results,[-1, self.num_units])
            match = tf.nn.xw_plus_b(results, self.match_proj_weight, self.match_proj_bias)
            match = tf.nn.elu(match)
            match = tf.reshape(match, [bsz, kMaxCheck, self.num_aspect])
            fw_rnn_cell = tf.nn.rnn_cell.GRUCell(self.num_units)
            bw_rnn_cell = tf.nn.rnn_cell.GRUCell(self.num_units)
            if training:
              fw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(fw_rnn_cell,input_keep_prob=0.8, output_keep_prob=0.8,state_keep_prob=0.8)
              bw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(bw_rnn_cell,input_keep_prob=0.8, output_keep_prob=0.8,state_keep_prob=0.8)
            
            
            outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, match,
                                              sequence_length = seglen,
                                              dtype=tf.float32)
            
            
          
            ret = tf.nn.xw_plus_b(tf.concat([states[0], states[1]],axis=1), self.out_weight, self.out_bias)
            ret = tf.squeeze(ret, axis=1, name=None if training else name)
            
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