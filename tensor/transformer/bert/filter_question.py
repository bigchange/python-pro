#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: filter_question.py
# Project: bert
# -----
# Copyright 2020 - 2019


import fire
import os
from os import walk
import codecs
import random


def parseQuery(inputPath,outPath):
  outp=open(outPath,"w")
  with open(inputPath,"r") as inp:
    for line in inp:
      line=line.strip()
      if not line:
        continue
      if line.find('怎')==-1 and line.find('么')==-1 and line.find("如何")==-1 and line.find("为何")==-1 and line.find("吗"):
        continue
      ss = line.split("\t")
      if len(ss)>6 or len(ss)==1:
        continue
      valid=True
      sameCs=set(list(unicode(ss[0].decode("utf8"))))
      for s in ss:
        nn=len(s)
        gotV=False
        for j in range(nn):
          if ord(s[j])>128:
            gotV=True
            break
        if not gotV:
          valid=False
          break
      if not valid:
        continue

      for i in range(1, len(ss)):
        cs=set(list(unicode(ss[i].decode("utf8"))))
        sameCs=cs.intersection(sameCs)
        if len(sameCs)==0:
          break
      if len(sameCs)==0:
        continue
      outp.write("%s\n"%(line))



if __name__ == "__main__":
    fire.Fire()

