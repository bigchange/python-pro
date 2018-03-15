#!/usr/bin/env python
# -*- coding:utf-8 -*-
# <<licensetext>>

import fire
import MySQLdb
import json
from dateutil.parser import parse
import time
import datetime
import csv
import tensorflow as tf
import random

MAX_HIST = 20
MAX_TITLE_CHARS = 32
MAX_SEARCH_CHARS = 20


# 字符级别的word ids
# vocab: 事先训练好的字符嵌入
def string2ids(str, vocab):
    if not vocab:
        return []
    chars = list(unicode(str.decode("utf8")))
    rets = []
    unk = 1
    for c in chars:
        if c not in vocab:
            rets.append(unk)
        else:
            rets.append(vocab[c])
    return rets


# 职位详情
class ProjectDetail:
    def __init__(self, pid, pubDate, modDate, title, location, minYear,
                 industry, salaryLow, salaryUp):
        self.pid = pid
        self.pubDate = pubDate
        self.modDate = modDate
        self.title = title
        self.location = location
        self.minYear = minYear
        self.industry = industry
        self.salaryLow = salaryLow
        self.salaryUp = salaryUp
        self.title_wids = None

    # 职位详情中title的wids
    def calc_title_wids(self, vocab):
        wids = string2ids(self.title, vocab)
        nn = len(wids)
        if nn < MAX_TITLE_CHARS:
            for i in range(nn, MAX_TITLE_CHARS):
                wids.append(0)
        elif nn > MAX_TITLE_CHARS:
            wids = wids[:MAX_TITLE_CHARS]
        assert (len(wids) == MAX_TITLE_CHARS)
        self.title_wids = wids


# 猎头和职位对
class UserProjPair:
    def __init__(self, uid, pid, utime, word=None, tag=1):
        self.uid = uid
        self.pid = pid
        self.utime = parse(utime)
        self.word = word  # 猎头搜索的关键词
        self.tag = tag  # 标签名： 0 for no action, 1 for search, 2 for click

    # pid是点击操作， 还是搜索操作
    def __str__(self):
        if self.word is None:
            return "Click\t" + str(self.pid) + "\t" + str(
                int(time.mktime(self.utime.timetuple())))
        else:
            return "Search\t" + self.word + "\t" + str(
                int(time.mktime(self.utime.timetuple())))


# 获取职位名称
def extract_title(jsonO):
    if "title" in jsonO:
        tt = jsonO['title']
        if tt is None:
            return None
        return tt.encode("utf8")
    else:
        return None


# 获取职位详情中的地点code
def extract_location(jsonO):
    if "addressCodes" in jsonO:
        addrs = jsonO['addressCodes']
        if addrs is None or len(addrs) == 0:
            return 0
        else:
            return int(addrs[0])
    else:
        return 0


# 获取行业code
def extact_industry(jsonO):
    if "industrialCodes" in jsonO:
        inds = jsonO['industrialCodes']
        if inds is None or len(inds) == 0:
            return 0
        else:
            indO = inds[0]
            if 'code' in indO:
                iv = indO['code']
                if iv is not None:
                    return int(iv)
                else:
                    return 0
    else:
        return 0


# 获取工作最小年限
def extract_minyear(jsonO):
    if "workExperience" in jsonO:
        weO = jsonO["workExperience"]
        if "required" in weO:
            rqO = weO["required"]
            if "workYearMin" in rqO:
                return int(rqO['workYearMin'])
            else:
                return 0
        else:
            return 0
    else:
        return 0


# 获取最高最低薪资要求
def extractSalary(jsonO):
    if "salary" in jsonO:
        sO = jsonO['salary']
        lo = 0
        up = 0
        if 'salaryLower' in sO:
            lo = int(sO['salaryLower'])
        if 'salaryUpper' in sO:
            up = int(sO['salaryUpper'])
        return lo, up
    else:
        return 0, 0


# 历史数据
class Hist:
    def __init__(self, proj, word):
        self.proj = proj
        self.word = word
        self.wids = None


# 训练的example封装类
class TrainEx:
    def __init__(self, uid, proj, label, clickTime):
        self.uid = uid
        self.proj = proj
        self.label = label
        ct = clickTime.date()
        self.clickTimeKey = ct.isoformat()
        self.hists = []

    # 添加历史proj
    def addHistByProj(self, proj):
        self.hists.append(Hist(proj, None))

    # 添加历史的搜索记录
    def addHistByWord(self, word, vocab):
        hist = Hist(None, word)
        # 搜索关键字需要做wids的获取
        wids = string2ids(word, vocab)
        nn = len(wids)
        if nn < MAX_SEARCH_CHARS:
            for i in range(nn, MAX_SEARCH_CHARS):
                wids.append(0)
        elif nn > MAX_SEARCH_CHARS:
            wids = wids[:MAX_SEARCH_CHARS]
        hist.wids = wids
        assert (len(hist.wids) == MAX_SEARCH_CHARS)
        self.hists.append(hist)

    # 历史数据中的所有pids
    def inHistPids(self):
        pids = []
        for h in self.hists:
            if h.word == None and h.proj is not None:
                pids.append(h.proj.pid)
        return set(pids)


# 转换成tfrecord
def convert2tf(ex, neg):
    proj = ex.proj
    label = ex.label
    assert (label != 0)
    # neg 是一种特殊的要求，设置后优先生效
    if neg is not None:
        proj = neg
        label = 0

    hist_title_wids = [0 for _ in range(MAX_TITLE_CHARS * MAX_HIST)]
    hist_salary_lo = [0 for _ in range(MAX_HIST)]
    hist_salary_up = [0 for _ in range(MAX_HIST)]
    hist_location = [0 for _ in range(MAX_HIST)]
    hist_min_year = [0 for _ in range(MAX_HIST)]
    hist_industry = [0 for _ in range(MAX_HIST)]
    hist_search_wids = [0 for _ in range(MAX_SEARCH_CHARS * MAX_HIST)]
    # 0 for no action, 1 for search, 2 for click
    hist_actions = [0 for _ in range(MAX_HIST)]
    nh = len(ex.hists)
    for i in range(nh):
        hist = ex.hists[i]
        if hist.word is None:
            hist_actions[i] = 1  # click
            hist_title_wids[i * MAX_TITLE_CHARS:(
                                                    i + 1) * MAX_TITLE_CHARS] = hist.proj.title_wids
            hist_salary_lo[i] = hist.proj.salaryLow
            hist_salary_up[i] = hist.proj.salaryUp
            hist_location[i] = hist.proj.location
            hist_min_year[i] = hist.proj.minYear
            hist_industry[i] = hist.proj.industry
        else:
            hist_actions[i] = -1
            hist_search_wids[i * MAX_SEARCH_CHARS:(
                                                      i + 1) * MAX_SEARCH_CHARS] = hist.wids
    assert (len(hist_search_wids) == MAX_HIST * MAX_SEARCH_CHARS)
    assert (len(hist_title_wids) == MAX_HIST * MAX_TITLE_CHARS)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "target":
                    tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "target_title":
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=proj.title_wids)),
                "target_lo_salary":
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=[proj.salaryLow])),
                "target_up_salary":
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=[proj.salaryUp])),
                "target_location":
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[proj.location])),
                "target_min_year":
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=[proj.minYear])),
                "target_industry":
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[proj.industry])),
                "hist_title_wids":
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=hist_title_wids)),
                "hist_salary_lo":
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=hist_salary_lo)),
                "hist_salary_up":
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=hist_salary_up)),
                "hist_location":
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=hist_location)),
                "hist_min_year":
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=hist_min_year)),
                "hist_search_wids":
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=hist_search_wids)),
                "hist_actions":
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=hist_actions)),
                "hist_industry":
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=hist_industry))
            }))
    return example


# 写入tfrecord
def writeToTfRecord(tfOut, ex, negProjs, ntimePos):
    example = convert2tf(ex, None)
    for _ in range(ntimePos):
        tfOut.write(example.SerializeToString())
    for nproj in negProjs:
        example = convert2tf(ex, nproj)
        tfOut.write(example.SerializeToString())
    return ntimePos + len(negProjs)


# 加载word嵌入字典
def load_vocab(path):
    ret = {}
    with open(path, "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line:
                continue
            ss = line.split("\t")
            assert (len(ss) == 2)
            idx = int(ss[1])
            word = ss[0]
            if word == "<UNK>" or word == "</s>":
                continue
            us = word.decode("utf8")
            assert (len(us) == 1)
            ret[us[0]] = idx
    return ret


def stats(trainOutPath, testOutPath, vocabPath, orderDataPath=None):
    vocab = load_vocab(vocabPath)
    hunterProjMap = {}
    hunterProjs = {}
    projectIds = set([])
    tfTrainOutput = tf.python_io.TFRecordWriter(trainOutPath)
    tfTestOutput = tf.python_io.TFRecordWriter(testOutPath)
    db = MySQLdb.connect(
        host="",
        port=3306,
        user="idmg",
        passwd="",
        db="ai_pipeline")
    cursor = db.cursor()
    # 处理订单数据 - 每个cid下的pids
    if orderDataPath is not None:
        with open(orderDataPath, 'rb') as csvfile:
            orderReader = csv.reader(csvfile, delimiter=',')
            for row in orderReader:
                if len(row) < 3:
                    continue
                hunterProjMap[str(row[0]) + "_" + str(row[1])] = 1
                cid = int(row[0])
                pid = int(row[1])
                hunterProjs.setdefault(cid, [])
                hunterProjs[cid].append(pid)
    else:
        cursor.execute("""SELECT project_id, c_id 
                    FROM order_data 
                    order by  updated_at
                    desc""")
        data = cursor.fetchall()
        for row in data:
            cid = row[1]
            pid = row[0]
            keystr = str(cid) + "_" + str(pid)
            hunterProjMap[keystr] = 1
            hunterProjs.setdefault(cid, [])
            hunterProjs[cid].append(pid)
    print("hunterProjMap size:%d" % (len(hunterProjMap)))
    # 处理职位详情信息
    cursor.execute(
        """SELECT project_id, project_detail,origin_updated_at,origin_created_at
                    FROM project_detail 
                    order by  updated_at
                    desc""")
    projMap = {}
    dateProjMap = {}
    data = cursor.fetchall()
    for row in data:
        pid = int(row[0])
        utime = row[2]
        ctime = row[3]
        jsonO = json.loads(row[1])
        title = extract_title(jsonO)
        lo, up = extractSalary(jsonO)
        ind = extact_industry(jsonO)
        minYear = extract_minyear(jsonO)
        loc = extract_location(jsonO)
        proj = ProjectDetail(pid, ctime, utime, title, loc, minYear, ind, lo,
                             up)
        proj.calc_title_wids(vocab)
        projMap[pid] = proj
        projPublishDate = datetime.date.fromtimestamp(ctime / 1000)
        projUpdateDate = datetime.date.fromtimestamp(utime / 1000)
        key1 = projPublishDate.isoformat()
        # 按每天统计project的量
        dateProjMap.setdefault(key1, [])
        dateProjMap[key1].append(proj)
        projectIds.add(pid)
    print("total projects:%d, proj map:%d" % (len(projectIds), len(projMap)))
    # 处理埋点数据 - 职位详细下的点击情况的筛选
    cursor.execute("""SELECT project_id, user_id, content,  updated_at
                  FROM action_event
                  WHERE action='click_c_project_detail'
                  order by  updated_at
                  desc""")
    data = cursor.fetchall()
    userProjMap = {}
    total = 0
    noContent = 0
    filtered = 0
    atime = None
    for row in data:
        pid = row[0]
        uid = row[1]
        content = row[2]
        if pid == 0:
            noContent += 1
        if not content or not content.strip():
            noContent += 1
            continue
        jsonO = json.loads(content)
        if "module" in jsonO:
            moduleName = jsonO['module']
            if moduleName == u'orders' or moduleName == u'calendar_view' or moduleName == u'history' or moduleName == u'project_notify' or moduleName == u'project_fail_recommend' or moduleName == u'project_create_success_recommend' or moduleName == u'im_project_list' or moduleName == u'project_assign':
                # 过滤掉已接单职位列表
                filtered += 1
                continue
            if 'createdAt' not in jsonO:
                filtered += 1
                continue
            createdAt = jsonO['createdAt']
            if createdAt is None or not createdAt:
                filtered += 1
                continue
            atime = createdAt.encode("utf8")

        userProjMap.setdefault(uid, [])
        userProjMap[uid].append(UserProjPair(uid, pid, atime))
        total += 1
    print("total valid:%d,noContent:%d, filtered:%d" % (total, noContent,
                                                       filtered))
    # 处理埋点数据 - 用户的搜索关键字处理
    cursor.execute("""SELECT project_id, user_id, content,  updated_at
                  FROM action_event
                  WHERE action='click_c_search'
                  order by  updated_at
                  desc""")
    data = cursor.fetchall()
    filtered = 0
    total = 0
    atime = None
    for row in data:
        pid = row[0]
        uid = row[1]
        content = row[2]
        word = None
        if not content or not content.strip():
            noContent += 1
            continue
        jsonO = json.loads(content)
        if "content" in jsonO:
            contentO = jsonO['content']
            if 'searchText' in contentO:
                word = contentO['searchText']
                if word is None or not word:
                    filtered += 1
                    continue
                else:
                    word = word.encode("utf8")
                if 'createdAt' not in jsonO:
                    filtered += 1
                    continue
                createdAt = jsonO['createdAt']
                if createdAt is None or not createdAt:
                    filtered += 1
                    continue
                atime = createdAt.encode("utf8")
            else:
                filtered += 1
                continue
        else:
            filtered += 1
        userProjMap.setdefault(uid, [])
        userProjMap[uid].append(UserProjPair(uid, pid, atime, word))
        total += 1
    print("total search valid:%d, filtered:%d" % (total, filtered))

    print("userProjMap size:%d" % (len(userProjMap)))
    projNotThereEx = 0
    totalEx = 0
    noNegs = 0
    totalPos = 0
    written = 0
    with open("debug.txt", "w") as outp:
        for k, v in userProjMap.iteritems():
            v = sorted(v, key=lambda a: a.utime)
            nv = len(v)
            v2 = []
            for i in range(nv):
                if i == 0:
                    v2.append(v[i])
                elif (v[i].utime - v[i - 1].utime).total_seconds() <= 2 or (
                                v[i].pid != 0 and v[i].pid == v[i - 1].pid
                ) or (v[i].word is not None and v[i].word == v[i - 1].word):
                    continue
                else:
                    v2.append(v[i])
            pidset = set([])
            v3 = []

            for i in range(len(v2), 0, -1):
                if v2[i - 1].word is None and v2[i - 1].pid != 0:
                    if v2[i - 1].pid in pidset:
                        continue
                    pidset.add(v2[i - 1].pid)
                    keystr = str(v2[i - 1].uid) + "_" + str(v2[i - 1].pid)
                    if keystr in hunterProjMap:
                        # print("got key in map:%s" % (keystr))
                        v2[i - 1].tag = 2
                elif v2[i - 1].word is None and v2[i - 1].pid == 0:
                    continue
                v3.append(v2[i - 1])

            v3.reverse()
            nn = len(v3)
            if nn == 0:
                continue

            donotUseForNeg = set([])
            if k in hunterProjs:
                donotUseForNeg = set(hunterProjs[k])

            for i in range(nn):
                if i > 0 and v3[i].word is None and v3[i].pid != 0:

                    if v3[i].pid not in projectIds:
                        projNotThereEx += 1
                        continue
                    if v3[i].pid not in projMap:
                        print("project not there????")
                        projNotThereEx += 1
                        continue
                    proj = projMap[v3[i].pid]
                    ex = TrainEx(k, proj, v3[i].tag, v3[i].utime)
                    for j in range(MAX_HIST):
                        cur = i - j - 1
                        if cur >= 0:
                            if v3[cur].word is None and v3[cur].pid != 0 and v3[cur].pid in projMap:
                                ex.addHistByProj(projMap[v3[cur].pid])
                            elif v3[cur].word is not None:
                                ex.addHistByWord(v3[cur].word, vocab)
                            else:
                                continue
                        else:
                            break

                    if len(ex.hists) > 0:
                        if ex.clickTimeKey not in dateProjMap:
                            noNegs += 1
                            continue
                        projs = dateProjMap[ex.clickTimeKey]
                        toFilters = pidset | ex.inHistPids() | donotUseForNeg
                        toFilters.add(ex.proj.pid)
                        projs = [p for p in projs if p.pid not in toFilters]
                        if len(projs) == 0:
                            noNegs += 1
                            continue
                        negProjs = []
                        if len(projs) <= 3:
                            negProjs = projs
                        else:
                            negProjs = random.sample(projs, 3)
                        tfOutput = tfTrainOutput
                        ntimePos = 1
                        if random.random() <= 0.034 and ex.label != 2:
                            tfOutput = tfTestOutput
                            # Only output 1 neg proj for test, make test accuracy more reliable
                            negProjs = negProjs[0:1]
                            ntimePos = 1
                        nex = writeToTfRecord(tfOutput, ex, negProjs, ntimePos)
                        totalEx += nex
                        totalPos += 1
                        if totalPos % 100 == 0:
                            print(
                                "project not there examples:%d, no neg exs:%d, total ex:%d(%d)"
                                % (projNotThereEx, noNegs, totalEx, totalPos))

    print("project not there examples:%d, no neg exs:%d, total ex:%d(%d)" %
          (projNotThereEx, noNegs, totalEx, totalPos))


def main():
    fire.Fire()


if __name__ == '__main__':
    main()
