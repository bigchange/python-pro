#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: split_baike.py
# Project: bert

# Copyright 2020 - 2018

import fire
import json
import re
import sentencepiece as spm
import random

MAX_TOKEN_NUM_PER_SENTENCE = 199


def split(content):
    rets = []
    nn = len(content)
    off = 0
    for i in range(nn):
        if content[i] == u'。' or content[i] == u'！' or content[i] == u'？' or content[i] == u'\n':
            if i > off:
                rets.append(content[off:i+1])
            off = i + 1
    if off < nn:
        rets.append(content[off:nn])
    return rets


def prepareLines(inputPath, outputP):
    processed = 0
    sentences = 0
    p = re.compile("(。)|(？)|(；)|[\n]", re.UNICODE)
    with open(inputPath, "r") as inp:
        with open(outputP, "w") as outp:
            for line in inp:
                line = line.strip()
                if not line:
                    continue
                processed += 1
                obj = json.loads(line)
                for k in obj:
                    v = obj[k]
                    if 'content' in v:
                        content = v['content']
                        ss = split(content)
                        for s in ss:
                            outp.write("%s\n" % (s.encode("utf8")))
                            sentences += 1
                if processed % 100000 == 0:
                    print("processed %d, got sentence:%d" %
                          (processed, sentences))
    print("processed %d, got sentence:%d" % (processed, sentences))


def filter4wv(inputPath, outputPath):
    with open(inputPath, "r") as inp:
        with open(outputPath, "w") as oup:
            for line in inp:
                line = line.strip()
                if not line or len(line) < 10:
                    continue
                line = line.lower()
                oup.write("%s\n" % (line))


def gen_sentence_features(sentence, seg):
    sentence = sentence.lower()
    ss = seg.EncodeAsIds(sentence)
    nt = len(ss)
    if nt > MAX_TOKEN_NUM_PER_SENTENCE:
        ss = ss[:MAX_TOKEN_NUM_PER_SENTENCE]
        nt = MAX_TOKEN_NUM_PER_SENTENCE
    for i in range(nt, MAX_TOKEN_NUM_PER_SENTENCE):
        ss.append(0)
    return ss, nt


def prepareW2V(trainPath,
               trainOutPath,
               modelPath):
    seg = spm.SentencePieceProcessor()
    seg.Load(modelPath)
    npos = 0
    nneg = 0
    skiped = 0
    processed = 0
    outp = open(trainOutPath, "w")
    with open(trainPath, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            processed += 1
            line = line.lower()
            if len(line) < 20:
                continue
            sentenceFeatures1, nt = gen_sentence_features(
                line, seg)
            if not sentenceFeatures1:
                skiped += 1
                continue
            ss = [str(sentenceFeatures1[i]) for i in range(nt)]
            outp.write("%s\n" % (" ".join(ss)))
            if processed % 100000 == 0:
                print("processed %d, skip:%d....." %
                      (processed, skiped))
    print("processed %d, skip:%d....." % (processed, skiped))
    outp.close()


def genBertTrain(inputPath, outputP):
    processed = 0
    sentences = 0
    p = re.compile("(。)|(？)|(；)|[\n]", re.UNICODE)
    history = []

    with open(inputPath, "r") as inp:
        with open(outputP, "w") as outp:
            for line in inp:
                line = line.strip()
                if not line:
                    continue
                processed += 1
                obj = json.loads(line)
                for k in obj:
                    v = obj[k]
                    if 'content' in v:
                        content = v['content']
                        ss = split(content)
                        nn = len(ss)
                        if nn > 2:
                            for i in range(nn):
                                if len(ss[i]) > 21:
                                    if len(history) < 10000:
                                        history.append(ss[i])
                                    else:
                                        if i < (nn-1) and len(ss[i+1]) > 21:
                                            outp.write("%s\x01%s\x011\n" % (
                                                ss[i].encode("utf8").lower(), ss[i+1].encode("utf8").lower()))
                                            idx = random.randint(0, 9000)
                                            outp.write("%s\x01%s\x010\n" % (
                                                ss[i].encode("utf8").lower(), history[idx].encode("utf8").lower()))
                                            sentences += 1
                                        history.pop(0)
                                        history.append(ss[i])

                if processed % 100000 == 0:
                    print("processed %d, got sentence:%d" %
                          (processed, sentences))
    print("processed %d, got sentence:%d" % (processed, sentences))


def main():
    fire.Fire()


if __name__ == '__main__':
    main()
