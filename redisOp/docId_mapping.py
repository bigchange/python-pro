import os

import redis

distinct_doc = {}

origin_doc = {}

block = "/Users/devops/workspace/shell/distinct_fingerprinter/block"

distinctDoc = "/Users/devops/workspace/shell/distinct_fingerprinter/distinct_docId_file"

originDocOne = "/Users/devops/workspace/shell/distinct_fingerprinter/docid_one"

originDocAnd = "/Users/devops/workspace/shell/distinct_fingerprinter/docid_two"

originDoc = "/Users/devops/workspace/shell/distinct_fingerprinter/docid"

dedupFile = "/Users/devops/workspace/shell/distinct_fingerprinter/dedup_store.1"

os.sys.path.append("/usr/local/lib/python2.7/site-packages/redis-2.10.5-py2.7.egg")

pool = redis.ConnectionPool(host='172.16.52.91', port=6379, db=14, password="DT:FA66AC61-C2F9-49F1-8A21-A14FCFD521E3")

r = redis.Redis(pool)

pipe = r.pipeline()

def reading_doc(dis, origin):
    with open(dis, "r") as file:
        index = 0
        for kv in [line.replace("\n", "").split("\t") for line in file]:
            if index == 0:
                print "line:" + line
                distinct_doc[kv[0]] = kv[1]
            else:
                distinct_doc[kv[0]] = kv[1]
            index += 1

            if index % 2004800 == 0:
                print "index:" + str(index)

    with open(origin, "r") as f:
        index = 0
        for line in f:
            lineFormat = line.replace("\n", "")
            if index == 0:
                print "line:" + lineFormat
            else:
                origin_doc[str(index)] = lineFormat

            index += 1

            if index % 2004800 == 0:
                print "index:" + str(index)

    print "distinc doc map size: " + str(len(distinct_doc))
    print "doc map size: " + str(len(origin_doc))

def writeDocIdRedis(distinct_doc):

    print "started writed redis .... "
    counter = 0
    redisListKey = "list:docId"
    size = len(distinct_doc)
    for index in range(1, size + 1, 1):
        name = str(index)
        value = distinct_doc[name]
        pipe.rpush(redisListKey, value)
        if counter % 204800 == 0:
            print "index:" + str(index)
            counter = 0
            print "index:" + str(index)
        else:
            counter += 1

    if counter > 0:
        print "index:" + str(index)


def readLineByLine(filePath):

    with open(filePath) as file:
        index = 0
        for line in file:
            if len(line.split("\t")) == 2:
                print "last index:" + str(index)
                break
            else:
                if index % 200000 == 0:
                    print("lequals 1 content:%s index:%s" % (line, str(index)))
            index += 1

def getBlockData(filePath):

    with open(filePath) as file:
        index = 0
        for line in file:
            if len(line.split("\t")) == 2:
                kv = line.replace("\n", "").split("\t")
                originId = origin_doc[kv[1]]
                newIndex = distinct_doc[originId]
                # print "originId:" + originId + ", newIndex :" + newIndex
            else:
                print "line length equals 1 content:" + line

            index += 1


# reading_doc(distinctDoc, originDoc)

# getBlockData(filePath=block)

# readLineByLine(dedupFile)

