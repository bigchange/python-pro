#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: parse_wiki.py
# Project: bert

# Copyright 2020 - 2019

import fire
import os
import json


def parse_weixin(path, outputPath):
    outp = open(outputPath, "w")
    totalDoc = 0
    skip = 0
    with open(path, "r") as inp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not ('content' in obj):
                skip += 1
                continue
            sentence = obj['content']
            sentence = sentence.replace(u'\n', u' ')
            sentence = sentence.replace(u"<br>", u"\n")
            sentence = sentence.replace(u"\r", u"")
            sentence = sentence.replace(u"&#x10;", u"\n")
            sentence = sentence.replace(u"<br/>", u"\n")
            sentence = sentence.replace(u";", u"\n")
            sentence = sentence.replace(u"；", u"\n")
            sentence = sentence.replace(u"。", u"\n")
            sentence = sentence.replace(u"o", u"\n")
            sentence = sentence.replace(u"　", u"")
            sentence = sentence.replace(u"&#8226", u"")
            totalDoc += 1
            if totalDoc % 10000 == 0:
                print("processed %d docs, skiped:%d...." % (totalDoc, skip))
            ss = sentence.split(u"\n")
            doc = []
            for s in ss:
                doc.append(s.encode("utf8"))
            if len(doc) <= 1:
                continue
            doc = doc[1:]
            nn = len(doc)
            nbatch = (nn-1)/6 + 1
            for i in range(nbatch):
                off = i*6
                end = off + 6
                if end > nn:
                    end = nn
                if (end-off) <= 2:
                    continue
                todos = doc[off:end]
                ss = "\t".join(todos)
                outp.write("%s\n" % (ss))
    outp.close()
    print("processed %d docs, skiped:%d...." % (totalDoc, skip))


def parseWiki(path, outputPath):
    outp = open(outputPath, "w")
    totalDoc = 0
    with open(path, "r") as inp:
        doc = []
        for line in inp:
            line = line.strip()
            if not line:
                continue
            if line.startswith("<doc "):
                pass
            elif line.startswith("</doc>"):
                totalDoc += 1
                if len(doc) <= 1:
                    continue
                doc = doc[1:]
                nn = len(doc)
                nbatch = (nn-1)/6 + 1

                for i in range(nbatch):
                    off = i*6
                    end = off + 6
                    if end > nn:
                        end = nn
                    if (end-off) <= 2:
                        continue
                    todos = doc[off:end]
                    ss = "\t".join(todos)
                    outp.write("%s\n" % (ss))
                doc = []
                if totalDoc % 10000 == 0:
                    print("processed %d docs...." % (totalDoc))
            else:
                line = line.replace('\t', ' ').replace(" ", " ").strip()
                if not line:
                    continue
                sentence = unicode(line.decode("utf8"))
                if len(sentence) <= 6:
                    continue
                sentence = sentence.replace(u"<br>", u"\n")
                sentence = sentence.replace(u"\r\n", u"\n")
                sentence = sentence.replace(u"&#x10;", u"\n")
                sentence = sentence.replace(u"<br/>", u"\n")
                sentence = sentence.replace(u";", u"\n")
                sentence = sentence.replace(u"；", u"\n")
                sentence = sentence.replace(u"。", u"\n")
                sentence = sentence.replace(u"o", u"\n")
                sentence = sentence.replace(u"　", u"")
                sentence = sentence.replace(u"&#8226", u"")
                ss = sentence.split(u"\n")
                for s in ss:
                    doc.append(s.encode("utf8"))
    outp.close()
    print("processed %d docs...." % (totalDoc))


if __name__ == "__main__":
    fire.Fire()
