#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: transformer.py
# Modify code based on openai's finetune-transformer-lm

import os
import time
import math
import json
import random
import argparse
import numpy as np
import tensorflow as tf


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def get_ema_if_exists(v, gvs):
    name = v.name.split(':')[0]
    ema_name = name+'/ExponentialMovingAverage:0'
    ema_v = [v for v in gvs if v.name == ema_name]
    if len(ema_v) == 0:
        ema_v = [v]
    return ema_v[0]


def get_ema_vars(*vs):
    if tf.get_variable_scope().reuse:
        gvs = tf.global_variables()
        vs = [get_ema_if_exists(v, gvs) for v in vs]
    if len(vs) == 1:
        return vs[0]
    else:
        return vs


def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x


def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable(
            "g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable(
            "bais", [n_state], initializer=tf.constant_initializer(0))
        g, b = get_ema_vars(g, b)
        return _norm(x, g, b, axis=axis)


def dropout(x, droprate):
    x = tf.nn.dropout(x, 1.0-droprate)
    return x


def mask_attn_weights_lm(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)  # Lower triangular part
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)  # w or very low value
    return w


def mask_attn_weights_for_length(w, mask):
    shapeList = shape_list(w)
    n = shapeList[-1]
    batch = shapeList[0]
    b = tf.expand_dims(mask, 1)
    b = tf.cast(tf.reshape(tf.tile(b, [1, n, 1]), [batch, 1, n, n]),tf.float32)
    # cm = tf.expand_dims(mask, 2)
    # cm = tf.reshape(tf.tile(cm, [1, 1, n]), [batch, 1, n, n])
    # b = rm*cm
    w = w*b + -1e9*(1-b)  # w or very low value
    return w


def _attn(q, k, v, droprate=0, scale=False, mask=None):
    w = tf.matmul(q, k)
    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))  # scaled dotproduct
    # use for LM, uncomment the next line
    # w = mask_attn_weights_lm(w)
    if mask is not None:
        w = mask_attn_weights_for_length(w, mask)

    w = tf.nn.softmax(w)

    w = dropout(w, droprate)

    a = tf.matmul(w, v)
    return a


def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]  # 把最后的一维拆成N部分相同大小的
    return tf.reshape(x, new_x_shape)


def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]  # N部分合并成一个
    return tf.reshape(x, new_x_shape)


def split_heads(x, n, k=False):
    if k:
        # batch,N-heads,embedding,seq-length
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        # batch,N-heads,seq-length,embedding
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def merge_heads(x):
    # 输入格式应该是:#batch,N-heads,seq-length,embedding
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=False,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    T = shape_list(inputs)[-1]
    N = shape_list(inputs)[-2]
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs


def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID'):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("bias", [nf], initializer=b_init)
        if rf == 1:  # faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(
                x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else:  # was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c


def attn(x, scope, n_state, n_head, droprate=0, scale=False, mask=None):
    assert n_state % n_head == 0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, droprate=droprate, scale=scale, mask=mask)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1)
        a = dropout(a, droprate)
        return a


def mlp(x, scope, n_state, droprate):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        h = tf.nn.relu(conv1d(x, 'c_fc', n_state, 1))
        h2 = conv1d(h, 'c_proj', nx, 1)
        h2 = dropout(h2, droprate)
        return h2

def block(x, n_head, scope, droprate=0, scale=False, mask=None):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, droprate=droprate,
                 scale=scale, mask=mask)
        n = norm(x+a, 'layer_norm_1')
        m = mlp(n, 'mlp', nx*4, droprate=droprate)
        h = norm(n+m, 'layer_norm_2')
        return h

# TransformerModel 的初始化
def model(n_head, n_layer, X, droprate=0, scale=False, mask=None):
    for layer in range(n_layer):
        X = block(X, n_head, 'transformer_%d' %
                  layer, droprate=droprate, scale=scale, mask=mask)
    return X
