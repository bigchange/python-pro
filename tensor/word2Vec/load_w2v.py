import numpy as np


# How to use word2Vec
def load_w2v(path, expectDim, checkUNK=True):
    fp = open(path, "r")
    print("load data from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    print ("total: ", total)
    dim = int(ss[1])
    print ("dim: ", dim)
    assert (dim == expectDim)
    ws = []
    # store each dim average value
    mv = [0 for i in range(dim)]

    second = -1
    for t in range(total):
        if ss[0] == '<UNK>':
            second = t
            print ("second :", second)
        line = fp.readline().strip()
        ss = line.split(" ")
        assert (len(ss) == (dim + 1))
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)
    for i in range(dim):
        mv[i] = mv[i] / total
    assert((not checkUNK) or (second != -1))
    # append two more token , maybe useless
    ws.append(mv)
    ws.append(mv)
    # exchange position with <UNK>, let <UNK> in position 1
    if checkUNK and second != 1:
        t = ws[1]
        print ("ws[1] :", ws[1])
        ws[1] = ws[second]
        ws[second] = t
    fp.close()
    print("loaded word2vec:%d" % (len(ws)))
    return np.asarray(ws, dtype=np.float32)


def main():
    load_w2v("/Users/devops/workspace/shell/e2e/matching/corpus/vec2.txt", 150)

if __name__ == "__main__":
    main()

