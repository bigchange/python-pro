#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: convert2tf.py
# Project: py
# -----
# Copyright 2020 - 2019


import tensorflow as tf
import fire
import sentencepiece as spm
import random
import json
import os
kMaxSeqLen = 127
kMaxXLen = 299
maxLen =0

def _convert2Example(seg, x, y,minXL=3, minYL=3, maxXL=kMaxXLen, maxYL=kMaxSeqLen, trimHead=False):
    global maxLen
    ex = None
    MASK = seg.GetPieceSize()
    SEP = MASK + 1
    wordsX = [SEP for _ in range(maxXL)]
    wordsY = [SEP for _ in range(maxYL)]
    ps = seg.EncodeAsIds(x)
    nn = len(ps)
    if nn >= maxXL:
        off = nn -maxXL
        if trimHead:
            ps = ps[off:]
        else:
            ps = ps[:nn]
        nn=maxXL
    for i in range(nn):
        wordsX[i] = ps[i] if ps[i] > 1 else MASK
    xLen = nn
    ps = seg.EncodeAsIds(y)
    nn = len(ps)
    if nn >= maxYL:
        nn = maxYL
        ps = ps[:nn]
    yLen = nn
    if xLen < minXL or yLen < minYL:
        # print("xlen %d < %d, or ylen %d < %d"%(xLen, minXL, yLen, minYL))
        return None
    if yLen > maxLen:
      maxLen = yLen
    for i in range(nn):
        wordsY[i] = ps[i] if ps[i] > 1 else MASK
    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                'x': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=wordsX)
                ),
                'y': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=wordsY)
                ),
                'xLen': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[xLen+1])
                ),
                'yLen': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[yLen+1])
                )
            }
        )
    )
    return ex




def countLength(spModelPath, inputPath):
    seg = spm.SentencePieceProcessor()
    seg.Load(spModelPath)
    
    lenTables=[0 for _ in range(300)]
    nall = 0
    with open(inputPath, "r") as inp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            summary= obj['title'].encode('utf8')
            content = obj['content'].encode('utf8')
            nall += 1
            ids = seg.EncodeAsIds(content)
            nn = len(ids)
            if nn >= 300:
                nn=299
            lenTables[nn]+=1

    for i in range(1, 300):
        if i<299:
            print("len %3d , occurred %6d times"%(i, lenTables[i]))
        else:
            print("len >= %3d , occurred %6d times"%(i, lenTables[i]))


def doGenDoubanTieba(spModelPath, inputPath, dataDir):
    global maxLen
    seg = spm.SentencePieceProcessor()
    seg.Load(spModelPath)
    train_writer = tf.python_io.TFRecordWriter(
        os.path.join(dataDir, 'tfrecord.train'))
    test_writer = tf.python_io.TFRecordWriter(
        os.path.join(dataDir, 'tfrecord.test'))
    writer = train_writer
    ntrain = 0
    ntest = 0
    nall = 0
    skipped = 0
    minXL=3
    minYL=3
    maxYL=95
    maxXL=199
    minXL=10
    minYL=5
    with open(inputPath, "r") as inp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            nall += 1
            ss =line.split("\t")
            summary= ss[1]
            content = ss[0]
            ex = _convert2Example(seg,content,summary,minXL, minYL, maxXL, maxYL, trimHead=True)
            if ex is None:
                skipped += 1
                continue
            writer = train_writer
            if random.random() < 0.01:
                writer = test_writer
                ntest += 1
            else:
                ntrain += 1
            writer.write(ex.SerializeToString())
            if nall % 10000 == 0:
                print("gen %d trains, %d tests, skipped=%d, maxLen=%d!" %
                      (ntrain, ntest, skipped,maxLen))
        print("gen %d trains, %d tests,skipped =%d, maxLen=%d!" %
              (ntrain, ntest, skipped,maxLen))


def doGen(spModelPath, inputPath, dataDir, forSummary=True):
    global maxLen
    seg = spm.SentencePieceProcessor()
    seg.Load(spModelPath)
    train_writer = tf.python_io.TFRecordWriter(
        os.path.join(dataDir, 'tfrecord.train'))
    test_writer = tf.python_io.TFRecordWriter(
        os.path.join(dataDir, 'tfrecord.test'))
    writer = train_writer
    ntrain = 0
    ntest = 0
    nall = 0
    skipped = 0
    minXL=3
    minYL=3
    maxYL=kMaxSeqLen
    maxXL=kMaxXLen
    if forSummary:
        minXL=30
        minYL=20
    with open(inputPath, "r") as inp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            nall += 1
            obj = json.loads(line)
            summary= obj['title'].encode('utf8')
            content = obj['content'].encode('utf8')
            ex = _convert2Example(seg,content,summary,minXL, minYL, maxXL, maxYL)
            if ex is None:
                skipped += 1
                continue
            writer = train_writer
            if random.random() < 0.01:
                writer = test_writer
                ntest += 1
            else:
                ntrain += 1
            writer.write(ex.SerializeToString())
            if nall % 10000 == 0:
                print("gen %d trains, %d tests, skipped=%d, maxLen=%d!" %
                      (ntrain, ntest, skipped,maxLen))
        print("gen %d trains, %d tests,skipped =%d, maxLen=%d!" %
              (ntrain, ntest, skipped,maxLen))


if __name__ == "__main__":
    fire.Fire()
