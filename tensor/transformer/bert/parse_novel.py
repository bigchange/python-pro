#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: parse_novel.py
# Project: bert
# -----
# Copyright 2020 - 2019

import fire
import os
from os import walk
import codecs
import random


def parseQuery(inputPrefix, numPart, inputPrefix2, numPart2, outPath):
    urlToQuery = {}
    with open(outPath, "w") as oup:
        for i in range(numPart):
            with open("%s%02d" % (inputPrefix, i), "r") as inp:
                for line in inp:
                    line = line.strip()
                    if not line:
                        continue
                    ss = line.split("\t")
                    u, q = ss[0], ss[1]
                    qs = urlToQuery.setdefault(u, set([]))
                    qs.add(q)
                    urlToQuery[u] = qs
        for i in range(numPart2):
            with open("%s%02d" % (inputPrefix2, i), "r") as inp:
                for line in inp:
                    line = line.strip()
                    if not line:
                        continue
                    ss = line.split("\t")
                    u, q = ss[0], ss[1]
                    qs = urlToQuery.setdefault(u, set([]))
                    qs.add(q)
                    urlToQuery[u] = qs
        count = 0
        for k, v in urlToQuery.iteritems():
            if len(v) > 2:
                vs = list(v)
                strs = "\t".join(vs)
                oup.write("%s\n" % (strs))
                count += 1
        print("got same meaning sessions:%d" % (count))


def _processNovel(lines, fp):
    nn = len(lines)
    if nn <= 100:
        return 0
    lines = lines[3:-3]
    nn = len(lines)
    gotL = 0
    gotC = 0
    doc = []
    ret = 0
    for i in range(nn):
        gotL += 1
        gotC += len(lines[i].decode("utf8"))
        doc.append(lines[i])
        if gotL > 4 or gotC > 180:
            gotL = 0
            gotC = 0
            docstr = "\t".join(doc)
            fp.write("%s\n" % (docstr))
            doc = []
            ret += 1
    return ret


kToChecks = ["吗", "么", "呢"]


def has_quest(line):
    for k in kToChecks:
        if line.find(k) != -1:
            return True
    return False


def parseNovel4Question(novelDir, outputPath):
    errF = 0
    total = 0
    gotLine = 0
    gotDoc = 0
    part = 0
    normWithQ = 0
    normNoQ = 0
    quest = 0
    posLabel = 0
    negLabel = 0
    skipP = 0
    skipN = 0
    fs = open(outputPath, "w")
    for (dirpath, dirnames, filenames) in walk(novelDir):
        for name in filenames:
            path = os.path.join(dirpath, name)
            with open(path, "r") as inp:
                content = inp.read()
                try:
                    content = codecs.decode(content, "gb18030").encode("utf8")
                except UnicodeDecodeError as ue:
                    print("================================>ignore %s" % (name))
                    errF += 1
                    continue
                total += 1
                lines = content.split("\n")
                lines = [l.strip() for l in lines]
                newss = []
                nonqs = []
                for line in lines:
                    pos = line.find("？")
                    if pos != -1:
                        line = line[:pos]
                        if not has_quest(line):
                            continue

                        pos1 = line.find("。")
                        pos2 = line.find("“")
                        if (pos1 == -1 and pos2 == -1) or random.random() < 0.5:
                            continue
                        quest += 1
                        if pos1 != -1 and pos2 != -1:
                            if pos1 < pos2:
                                line = line[pos2+3:]
                                newss.append(line)
                            else:
                                line = line[pos1+3:]
                                newss.append(line)
                        elif pos2 != -1:
                            line = line[pos2+3:]
                            newss.append(line)
                        elif pos1 != -1:
                            line = line[pos1+3:]
                            newss.append(line)
                        else:
                            assert(False)
                    else:
                        pos = line.find("。")
                        if pos != -1:
                            line = line[:pos]
                            hasQ = has_quest(line)
                            if hasQ or random.random() < 0.13:
                                pos1 = line.find("。")
                                pos2 = line.find("“")
                                if pos1 == -1 and pos2 == -1:
                                    continue
                                if hasQ:
                                    normWithQ += 1
                                else:
                                    normNoQ += 1
                                if pos1 != -1 and pos2 != -1:
                                    if pos1 < pos2:
                                        line = line[pos2+3:]
                                        nonqs.append(line)
                                    else:
                                        line = line[pos1+3:]
                                        nonqs.append(line)
                                elif pos2 != -1:
                                    line = line[pos2+3:]
                                    nonqs.append(line)
                                elif pos1 != -1:
                                    line = line[pos1+3:]
                                    nonqs.append(line)
                                else:
                                    assert(False)

                for s in newss:
                    if s.find("”") != -1 or s.find("！") != -1 or len(s) < 12 or len(s) > 128:
                        skipP += 1
                        continue
                    posLabel += 1
                    fs.write("1\t%s\n" % (s))
                for s in nonqs:
                    if s.find("”") != -1 or s.find("！") != -1 or len(s) < 12 or len(s) > 128:
                        skipN += 1
                        continue
                    negLabel += 1
                    fs.write("0\t%s\n" % (s))

    print("processed:%d, error:%d, questions:%d, non-question=%d, normWithQ=%d, normNoQ=%d,quest=%d,skipP=%d, skipN=%d " %
          (total, errF, posLabel, negLabel, normWithQ, normNoQ, quest, skipP, skipN))


def parseNovel(novelDir, outputPrefix):
    errF = 0
    total = 0
    gotLine = 0
    gotDoc = 0
    part = 0
    fs = []
    for i in range(10):
        fs.append(open("%s_%d.txt" % (outputPrefix, i), "w"))
    for (dirpath, dirnames, filenames) in walk(novelDir):
        for name in filenames:
            path = os.path.join(dirpath, name)
            with open(path, "r") as inp:
                content = inp.read()
                try:
                    content = codecs.decode(content, "gb18030").encode("utf8")
                except UnicodeDecodeError as ue:
                    print("================================>ignore %s" % (name))
                    errF += 1
                    continue
                total += 1
                lines = content.split("\n")
                lines = [l.strip() for l in lines]
                content = "。".join(lines)
                # print(content)
                ss = content.split("。")
                ss = [s.replace("　", "").replace('\t', " ").strip()
                      for s in ss]
                newss = []
                for s in ss:
                    if len(s) > 4 and s.find("本书来自") == -1 and s.find("免费电子书") == -1:
                        newss.append(s)
                nn = len(newss)
                if nn < 500:
                    errF
                    continue
                gotLine += nn
                part = gotDoc / 500000
                gotDoc += _processNovel(newss, fs[part])
    for i in range(10):
        fs[i].close()

    print("processed:%d, error:%d, got lines:%d, docs=%d" %
          (total, errF, gotLine, gotDoc))


def prepare4parrel(inputPath, outputPath):
    outp = open(outputPath, "w")
    with open(inputPath, "r") as inp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            ss = line.split("\t$$$\t")
            nn = len(ss)
            if nn < 2:
                continue
            i = 0
            while i < nn-1:
                if len(ss[i]) < 30:
                    i += 1
                    continue
                outp.write("%s\t$$$\t%s\n" % (ss[i], ss[i+1]))
                i += 2

    outp.close()


def _extractChats(ss):
    rets = []
    lastPos = -1
    nowChat = []
    for i in range(len(ss)):
        psp = ss[i].rfind("“")
        sp = ss[i].find("“")
        if psp != sp and sp != -1 and psp != -1:
            continue
        ep = sp
        if sp != -1:
            ep = ss[i].find("”", sp)
        if sp != ep and ep != -1 and (ep-sp) > 6:
            sentence = ss[i][sp+3:ep]
            # print("got sentences:%s"%(sentence))
            if (i-lastPos) > 1:
                if len(nowChat) > 1:
                    rets.append(nowChat)
                nowChat = []
            nowChat.append(sentence)
            lastPos = i
        elif sp != ep and ep != -1:
            if len(nowChat) > 1:
                rets.append(nowChat)
            nowChat = []
            lastPos = i

    if len(nowChat) > 1:
        rets.append(nowChat)
    return rets


def extractChat(novelDir, outputPath):
    errF = 0
    total = 0
    gotLine = 0
    gotDoc = 0
    part = 0
    outp = open(outputPath, "w")
    for (dirpath, dirnames, filenames) in walk(novelDir):
        for name in filenames:
            path = os.path.join(dirpath, name)
            with open(path, "r") as inp:
                content = inp.read()
                try:
                    content = codecs.decode(content, "gb18030").encode("utf8")
                except UnicodeDecodeError as ue:
                    print("================================>ignore %s" % (name))
                    errF += 1
                    continue
                total += 1
                lines = content.split("\n")
                lines = [l.strip() for l in lines]
                # content = "。".join(lines)
                # print(content)
                # ss = content.split("。")
                ss = [s.replace("　", "").replace('\t', " ").strip()
                      for s in lines]
                newss = []
                for s in ss:
                    if len(s) > 4 and s.find("本书来自") == -1 and s.find("免费电子书") == -1:
                        newss.append(s)
                nn = len(newss)
                if nn < 500:
                    errF
                    continue
                gotLine += nn
                chats = _extractChats(newss)
                gotDoc += len(chats)
                for ch in chats:
                    outp.write("%s\n" % "\t$$$\t".join(ch))
    outp.close()
    print("processed:%d, error:%d, got lines:%d, docs=%d" %
          (total, errF, gotLine, gotDoc))


if __name__ == "__main__":
    fire.Fire()
