import os

import redis

distinct_doc = {}

origin_doc = {}

block = "/Users/devops/workspace/shell/distinct_fingerprinter/block"

distinct_doc_path = "/Users/devops/workspace/shell/distinct_fingerprinter/distinct_docId_file"

origin_doc_one_path = "/Users/devops/workspace/shell/distinct_fingerprinter/docid_one"

origin_doc_two_path = "/Users/devops/workspace/shell/distinct_fingerprinter/docid_two"

origin_doc_path = "/Users/devops/workspace/shell/distinct_fingerprinter/docid"

dedup_file = "/Users/devops/workspace/shell/distinct_fingerprinter/dedup_store.1"

os.sys.path.append("/usr/local/lib/python2.7/site-packages/redis-2.10.5-py2.7.egg")

pool = redis.ConnectionPool(host='xxx', port=6390, db=14, password="DT:xx")

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
            line_format = line.replace("\n", "")
            if index == 0:
                print "line:" + line_format
            else:
                origin_doc[str(index)] = line_format

            index += 1

            if index % 2004800 == 0:
                print "index:" + str(index)

    print "distinc doc map size: " + str(len(distinct_doc))
    print "doc map size: " + str(len(origin_doc))


def write_docid_redis(distinct_doc):
    print "started writed redis .... "
    counter = 0
    redis_list_key = "list:docId"
    size = len(distinct_doc)
    for index in range(1, size + 1, 1):
        name = str(index)
        value = distinct_doc[name]
        pipe.rpush(redis_list_key, value)
        if counter % 204800 == 0:
            print "index:" + str(index)
            counter = 0
            print "index:" + str(index)
        else:
            counter += 1

    if counter > 0:
        print "index:" + str(index)


def read_line_by_line(file_path):
    with open(file_path) as file:
        index = 0
        for line in file:
            if len(line.split("\t")) == 2:
                print "last index:" + str(index)
                break
            else:
                if index % 200000 == 0:
                    print("lequals 1 content:%s index:%s" % (line, str(index)))
            index += 1


def get_block_data(file_path):
    with open(file_path) as file:
        index = 0
        for line in file:
            if len(line.split("\t")) == 2:
                kv = line.replace("\n", "").split("\t")
                origin_id = origin_doc[kv[1]]
                new_index = distinct_doc[origin_id]
                # print "originId:" + originId + ", newIndex :" + newIndex
            else:
                print "line length equals 1 content:" + line

            index += 1

# reading_doc(distinctDoc, originDoc)

# getBlockData(filePath=block)

# readLineByLine(dedupFile)
