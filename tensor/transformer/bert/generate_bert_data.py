#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: generate_bert_data.py

# Copyright 2020 - 2018

import fire
import json
import random

def doSampling(trainPrefixPath, outPath, partFrom=0, partEnd=49):
    processed = 0
    emptyS = 0
    tooShort = 0
    gotS = 0    
    fout = open(outPath, "w")
    history = []
    for i in range(partFrom, partEnd + 1):
        with open("%s-%05d" % (trainPrefixPath, i), "r") as fp:
            for line in fp.readlines():
                line = line.strip()
                if not line:
                    continue
                processed += 1
                ss = line.split("\t", 2)
                data = None
                try:
                    data = json.loads(ss[1])
                except UnicodeDecodeError as uex:
                    continue
                sentence = data['job_description'].encode("utf8").lower()
                if not sentence:
                    emptyS += 1
                else:
                    sentence = sentence.replace("<br/>", "\n")
                    sentence = sentence.replace("\r\n", "\n")
                    sentence = sentence.replace("&#x10;", "\n")
                    sentences = sentence.split("\n")
                    localSentences = []
                    for s in sentences:
                        us = s.decode("utf8")
                        ss = us.split(u"。；！？")
                        for a in ss:
                            theStr = a.encode("utf8")
                            # print(theStr)
                            localSentences.append(theStr)
                    nn = len(localSentences)
                    if nn > 2:
                        for i in range(nn):
                            if len(localSentences[i]) > 21:
                                if len(history) < 10000:
                                    history.append(localSentences[i])
                                else:
                                    if i < (nn-1) and len(localSentences[i+1]) > 21:
                                        fout.write("%s\x01%s\x011\n" % (
                                            localSentences[i], localSentences[i+1]))
                                        idx = random.randint(0, 9000)
                                        gotS +=1
                                        fout.write("%s\x01%s\x010\n" % (
                                            localSentences[i], history[idx]))
                                    history.pop(0)
                                    history.append(localSentences[i])
                if processed % 20000 == 0:
                    print(
                        "processed %d, emptySentence:%d, too short sentence:%d, valid sentences:%d....."
                        % (processed, emptyS, tooShort, gotS))
    print(
        "processed %d, emptySentence:%d, too short sentence:%d, valid sentences:%d....."
        % (processed, emptyS, tooShort, gotS))
    fout.close()


def main():
    fire.Fire()


if __name__ == '__main__':
    main()
