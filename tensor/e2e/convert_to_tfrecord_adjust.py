#!/usr/bin/env python
import os
import tensorflow as tf
import w2v
import fire
import raseg
import random
import json
import base64

MAX_TOKEN_NUM_PER_SENTENCE = 120
maxTokens = 0
MAX_EXS_PER_CLASS = 5500


def load_title_dict(path, titleVocab):
    with open(path, "r") as fp:
        for line in fp.readlines():
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            if not line:
                continue
            ss = line.split("\t")
            if not ss[0].strip():
                continue
            cid = int(ss[0])
            title = ss[1].lower()
            if ss[3].strip():
                title = ss[3].strip().lower()
            # print("title:[%s],id:[%d]"%(title,cid))
            titleVocab[title] = cid


def gen_sentence_features(sentence, vocab, seg):
    global maxTokens
    sentence = sentence.lower()
    ss = seg.tokenize_string(sentence, False)
    nt = 0
    ret = []
    for s in ss:
        s = s.strip()
        if not s:
            continue
        idx = vocab.GetWordIndex(s)
        nt += 1
        if nt <= MAX_TOKEN_NUM_PER_SENTENCE:
            ret.append(idx)
        else:
            break
    if nt > maxTokens:
        maxTokens = nt
    nt = len(ret)
    for i in range(nt, MAX_TOKEN_NUM_PER_SENTENCE):
        ret.append(0)
    return ret


class EduExperience:
    def __init__(self, edustrs, off):
        self.school = int(edustrs[off * 4])
        self.degree = int(edustrs[off * 4 + 1])
        self.start = float(edustrs[off * 4 + 2])
        self.major = int(edustrs[off * 4 + 3])


class WorkExperience:
    def __init__(self, workstrs, off):
        self.org = int(workstrs[off * 6])
        self.start = float(workstrs[off * 6 + 1])
        self.duaration = float(workstrs[off * 6 + 2])
        self.job = int(workstrs[off * 6 + 3])
        self.orgId = int(workstrs[off * 6 + 4])
        if workstrs[off * 6 + 5] != "None":
            self.desc = base64.b64decode(workstrs[off * 6 + 5])
        else:
            self.desc = ""


def convert(trainPath,
            trainOutPath,
            testOutPath,
            vocabPath,
            titleVobPath,
            partFrom=0,
            partEnd=9,
            testRatio=0.02):
    vocab = w2v.Word2vecVocab()
    vocab.Load(vocabPath)
    writerTrain = tf.python_io.TFRecordWriter(trainOutPath)
    writerTest = tf.python_io.TFRecordWriter(testOutPath)
    raseg.init_config("/var/local/seg/conf/qsegconf.ini")
    seg = raseg.ImTokenizer()
    titleVocab = {}
    load_title_dict(titleVobPath, titleVocab)
    numTag = len(titleVocab) + 1
    npos = 0
    nneg = 0
    processed = 0

    for i in range(partFrom, partEnd + 1):
        with open("%s/part-r-%05d" % (trainPath, i), "r") as fp:
            for line in fp.readlines():
                line = line.strip()
                if not line:
                    continue
                processed += 1
                ss = line.split("\t")
                assert (len(ss) == 8)
                title = ss[0].lower()
                if title == '网络':
                    title = '网络工程师'
                if title not in titleVocab:
                    print("[%s] not there!! " % (title))
                    continue

                target = titleVocab[title]
                target_orgId = int(ss[1])
                gender = int(ss[2])
                age = int(ss[3])
                location = int(ss[4])
                edustrs = ss[5].split(" ")
                assert(len(edustrs) == 12)
                edu_expr1 = EduExperience(edustrs, 0)
                edu_expr2 = EduExperience(edustrs, 1)
                edu_expr3 = EduExperience(edustrs, 2)

                workstrs = ss[6].split(" ")
                assert(len(workstrs) == 18)
                work_expr1 = WorkExperience(workstrs, 0)
                workTokens = gen_sentence_features(work_expr1.desc,
                                                   vocab, seg)
                work_expr2 = WorkExperience(workstrs, 1)
                workTokens += gen_sentence_features(work_expr2.desc,
                                                    vocab, seg)
                work_expr3 = WorkExperience(workstrs, 2)
                workTokens += gen_sentence_features(work_expr3.desc,
                                                    vocab, seg)

                projstrs = ss[7].split(" ")
                assert(len(projstrs) == 3)
                proj1 = ""
                if projstrs[0] != 'None':
                    proj1 = base64.b64decode(projstrs[0])
                proj2 = ""
                if projstrs[1] != 'None':
                    proj2 = base64.b64decode(projstrs[1])
                proj3 = ""
                if projstrs[2] != 'None':
                    proj3 = base64.b64decode(projstrs[2])

                projTokens = gen_sentence_features(proj1,
                                                   vocab, seg)
                projTokens += gen_sentence_features(proj2,
                                                    vocab, seg)
                projTokens += gen_sentence_features(proj3,
                                                    vocab, seg)

                assert(len(workTokens) == (3 * MAX_TOKEN_NUM_PER_SENTENCE))
                example = tf.train.Example(features=tf.train.Features(feature={
                    "target": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[target])),
                    "target_orgId": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[target_orgId])),
                    "gender": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[gender])),
                    "age": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[age])),
                    "location": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[location])),
                    "education_schools": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[edu_expr1.school, edu_expr2.school, edu_expr3.school])),
                    "education_degrees": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[edu_expr1.degree, edu_expr2.degree, edu_expr3.degree])),
                    "education_starts": tf.train.Feature(float_list=tf.train.FloatList(
                        value=[edu_expr1.start, edu_expr2.start, edu_expr3.start])),
                    "education_majors": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[edu_expr1.major, edu_expr2.major, edu_expr3.major])),

                    "work_expr_orgs": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[work_expr1.org, work_expr2.org, work_expr3.org])),
                    "work_expr_starts": tf.train.Feature(float_list=tf.train.FloatList(
                        value=[work_expr1.start, work_expr2.start, work_expr3.start])),
                    "work_expr_durations": tf.train.Feature(float_list=tf.train.FloatList(
                        value=[work_expr1.duaration, work_expr2.duaration, work_expr3.duaration])),
                    "work_expr_jobs": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[work_expr1.job, work_expr2.job, work_expr3.job])),
                    "work_expr_orgIds": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[work_expr1.orgId, work_expr2.orgId, work_expr3.orgId])),
                    "work_expr_descs": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=workTokens)),
                    "proj_expr_descs": tf.train.Feature(int64_list=tf.train.Int64List(
                        value=projTokens)),
                }))
                if random.random() <= testRatio:
                    writerTest.write(example.SerializeToString())
                    nneg += 1
                else:
                    writerTrain.write(example.SerializeToString())
                    npos += 1
                if processed % 200 == 0:
                    print("processed %d, neg:%d, pos:%d....." %
                          (processed, nneg, npos))
    print("max len of sentences:%d" % (maxTokens))
    writerTrain.close()
    writerTest.close()


if __name__ == '__main__':
    fire.Fire()
