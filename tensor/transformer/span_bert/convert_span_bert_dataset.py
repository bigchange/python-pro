#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: convert_span_bert_dataset.py
# Project: span_bert
# Copyright 2020 - 2019

import json
import random

import fire
import numpy as np
import sentencepiece as spm
import tensorflow as tf

kMaxSeqLen = 199
# mask的最大数量
kMaxTarget=26

def _mask_seq(SPSIZE, X, numMaxTarget):
  nn=len(X)
  targets=[]
  targetIndexes=[]
  spansLeft=[]
  spansRight=[]
  # wordLength sample weight
  lensWeights=[0.1,0.15,0.2,0.2,0.2,0.15]
  # 最长wordlength = 6, 最小wordlength=1
  lensVal =[1,2,3,4,5,6]
  MASK = SPSIZE+1
  #先打乱顺序
  perms =np.random.permutation(nn/4)
  # print("perms=%r"%(perms))
  masked =[False for _ in range(nn)]
  nm = 0
  for p in perms:
    # print("p=%d"%(p))
    off = int(p*4)
    if nm >=numMaxTarget:
      break
    drawLen=int(np.random.choice(lensVal, 1,
              p=lensWeights))
    # print("drawLen=%d"%(drawLen))
    if nm + drawLen >numMaxTarget:
      continue
    if drawLen <4:
      cands=[i for i in range(5-drawLen)]
      off += int(np.random.choice(cands,1))
    valid = True
    for i in range(drawLen):
      if masked[i+off] or (i+off==nn-1):
        valid = False
        break
    if valid:
      for i in range(drawLen):
        masked[i+off]=True
      nm +=drawLen
  
  start=-1
  for i in range(nn):
    if masked[i]:
      if start==-1:
        start=i
    else:
      if start!=-1:
        for k in range(start,i):
            targets.append(X[k])
            targetIndexes.append(k+1)
            spansLeft.append(start)
            spansRight.append(i+1)
            if random.random() <= 0.8:
                  X[k] = MASK
            else:
                  if random.random() <= 0.5:
                      # random one
                      X[k] = random.randint(1, SPSIZE-1)
        start =-1
  assert(start ==-1)
  if len(targets) > numMaxTarget :
    print("num targets:%d"%(len(targets)))
    assert(False)
  return targets, targetIndexes,spansLeft, spansRight


def doConvert(spModelPath, inputPath,trainOutputPath, testOutputPath):
    seg = spm.SentencePieceProcessor()
    seg.Load(spModelPath)
    nall = 0
    valid =0
    maxSpSize = seg.GetPieceSize()
    MASK = maxSpSize+1
    SEP = maxSpSize
    tables=[0 for _ in range(kMaxTarget)]
    train_writer = tf.python_io.TFRecordWriter(trainOutputPath)
    test_writer = tf.python_io.TFRecordWriter(testOutputPath)
    writer = train_writer
    ntrain = ntest =0
    print("input:%s,  train:%s, test:%s"%(inputPath, trainOutputPath, testOutputPath))
    with open(inputPath, "r") as inp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            desc= obj['des'].encode('utf8')
            desc = desc.lower()
            nall += 1
            ids = seg.EncodeAsIds(desc)
            nn = len(ids)
            if nn<20:
              continue
            valid +=1
            if nn > kMaxSeqLen:
                nn=kMaxSeqLen
                ids=ids[:nn]
            mxTargets=int(nn*0.15)
            if mxTargets> kMaxTarget:
              mxTargets = kMaxTarget
            targets, targetIndexes,spansLeft, spansRight=_mask_seq(maxSpSize, ids, mxTargets)
            for _ in range(nn, kMaxSeqLen):
              ids.append(0)
            if targets is None or len(targets) ==0:
              continue
            nt = len(targets)
            tables[nt-1]+=1
            for i in range(nt, kMaxTarget):
              targets.append(0)
              targetIndexes.append(0)
              spansLeft.append(0)
              spansRight.append(0)
            nt = len(targets)
            assert(nt == kMaxTarget)
            assert(nt==len(targetIndexes))
            assert(nt==len(spansLeft))
            assert(nt==len(spansRight))
            assert(len(ids)==kMaxSeqLen) 
            ex = tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'x': tf.train.Feature(
                          int64_list=tf.train.Int64List(value=ids)
                      ),
                      'y': tf.train.Feature(
                          int64_list=tf.train.Int64List(value=targets)
                      ),
                      # 文本的实际有效长度
                      'len': tf.train.Feature(
                          int64_list=tf.train.Int64List(value=[nn])
                      ),
                      'spansLeft': tf.train.Feature(
                          int64_list=tf.train.Int64List(value=spansLeft)
                      ),
                      'spansRight': tf.train.Feature(
                          int64_list=tf.train.Int64List(value=spansRight)
                      ),
                      'labelIndex': tf.train.Feature(
                          int64_list=tf.train.Int64List(value=targetIndexes)
                      )
                  }
              )
            )
            writer = train_writer
            if random.random() < 0.003:
                writer = test_writer
                ntest += 1
            else:
                ntrain += 1
            writer.write(ex.SerializeToString())
            if (nall % 10000)==0:
              print("processed  %d, valid:%d , train/test : %d/%d......."%(nall, valid,ntrain, ntest))
    print("processed  %d, valid:%d , train/test : %d/%d......."%(nall, valid,ntrain, ntest))
    # 每个number数量的targets出现次数
    for i in range(kMaxTarget):
      print("targets %02d , occurs %07d"%(i+1, tables[i]))
      
if __name__ == '__main__':
  fire.Fire()
           
