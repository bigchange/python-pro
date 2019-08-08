#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: save4serving.py
# Project: bert

# Copyright 2020 - 2019

import tensorflow as tf
import fire

def doSave(modelPath,savePath, version):
  

  # 如果训练后直接保存，跳过restore的过程 
  with tf.Session() as sess:
    graph= tf.get_default_graph()
    metaPath="%s.meta"%(modelPath)
    print("meta path:%s"%(metaPath))
    saver = tf.train.import_meta_graph(metaPath)
    saver.restore(sess, modelPath)
    
    # 获取节点
    sentenceN = graph.get_tensor_by_name('sentence:0')
    lengthN = graph.get_tensor_by_name('totalLength:0')
    alengthN = graph.get_tensor_by_name('alength:0')
    dropoutN = graph.get_tensor_by_name('dropout:0')
    reprN = graph.get_tensor_by_name('repr:0')
    
    # build info
    
    sentence_info = tf.saved_model.utils.build_tensor_info(sentenceN)
    length_info = tf.saved_model.utils.build_tensor_info(lengthN)
    alength_info = tf.saved_model.utils.build_tensor_info(alengthN)
    dropout_info = tf.saved_model.utils.build_tensor_info(dropoutN)
    repr_info = tf.saved_model.utils.build_tensor_info(reprN)
    
    # 生成signature
    pred_signature = (tf.saved_model.signature_def_utils.build_signature_def(inputs={'sentence': sentence_info, 'totalLength': length_info,"aLength":alength_info,"dropout":dropout_info}, outputs={'repr':repr_info}, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    
    
    # build & save
    builder = tf.saved_model.builder.SavedModelBuilder('%s/%d'%(savePath, version)) # version为代表版本的正整数
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={'bert_repr': pred_signature}, main_op=tf.tables_initializer(), strip_default_attrs=True)
    builder.save()


if __name__ == "__main__":
    fire.Fire()