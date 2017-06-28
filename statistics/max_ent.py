# coding=utf-8

from collections import defaultdict
import math


# GIS训练算法最大熵模型
class MaxEnt(object):
    def __init__(self):
        self.feats = defaultdict(int)
        self.trainset = []
        self.labels = set()
        self.w = []

    def load_data(self, file):
        for line in open(file):
            fields = line.strip().split()
            # at least two columns
            if len(fields) < 2:
                continue
            # the first column is label
            label = fields[0]
            self.labels.add(label)
            for f in set(fields[1:]):
                # (label,f) tuple is feature
                self.feats[(label, f)] += 1
            self.trainset.append(fields)
        print "self.feats -> " + str(self.feats)

    def _initparams(self):
        self.size = len(self.trainset)
        # M param for GIS training algorithm
        self.M = max([len(record) - 1 for record in self.trainset])
        print "self.M -> " + str(self.M)
        self.ep_ = [0.0] * len(self.feats)
        print "self.EP_ -> " + str(self.ep_)
        for i, f in enumerate(self.feats):
            # calculate feature expectation on empirical distribution
            self.ep_[i] = float(self.feats[f]) / float(self.size)   # 1/N
            # each feature function correspond to id
            self.feats[f] = i
        # init weight for each feature
        self.w = [0.0] * len(self.feats)
        print "self.w -> " + str(self.w)
        self.lastw = self.w

    """calculate p(y|x) """
    def calprob(self, features):
        print "features -> " + str(features)
        wgts = [(self.probwgt(features, l), l) for l in self.labels]
        print "wgts -> " + str(wgts)
        Z = sum([w for w, l in wgts])
        print "Z -> " + str(Z)
        prob = [(w / Z, l) for w, l in wgts]
        print "prob -> " + str(prob)
        return prob

    """sum in one label of each feature """
    def probwgt(self, features, label):
        wgt = 0.0
        for f in features:
            if (label, f) in self.feats:
                wgt += self.w[self.feats[(label, f)]]
        return math.exp(wgt)

    """
    calculate feature expectation on model distribution
    """

    def Ep(self):
        ep = [0.0] * len(self.feats)
        for record in self.trainset:
            features = record[1:]
            # calculate p(y|x)
            prob = self.calprob(features)
            for f in features:
                for w, l in prob:
                    # only focus on features from training data.
                    if (l, f) in self.feats:
                        # get feature id
                        idx = self.feats[(l, f)]
                        # sum(1/N * f(y,x)*p(y|x)), p(x) = 1/N
                        ep[idx] += w * (1.0 / self.size)
        return ep

    def _convergence(self, lastw, w):
        for w1, w2 in zip(lastw, w):
            if abs(w1 - w2) >= 0.01:
                return False
        return True

    """
    train data ：
    Outdoor Sunny Happy
    Outdoor Sunny Happy Dry
    Outdoor Sunny Happy Humid
    Outdoor Sunny Sad Dry
    Outdoor Sunny Sad Humid
    Outdoor Cloudy Happy Humid
    Outdoor Cloudy Happy Humid
    Outdoor Cloudy Sad Humid
    Outdoor Cloudy Sad Humid
    Indoor Rainy Happy Humid
    Indoor Rainy Happy Dry
    Indoor Rainy Sad Dry
    Indoor Rainy Sad Humid
    Indoor Cloudy Sad Humid
    Indoor Cloudy Sad Humid
    """

    def train(self, max_iter=1000):
        self._initparams()
        for i in range(max_iter):
            print 'iter %d ...' % (i + 1)
            # calculate feature expectation on model distribution
            self.ep = self.Ep()
            self.lastw = self.w[:]
            for i, w in enumerate(self.w):
                delta = 1.0 / self.M * math.log(self.ep_[i] / self.ep[i])
                # update w
                self.w[i] += delta
            print self.w
            # test if the algorithm is convergence
            if self._convergence(self.lastw, self.w):
                break

    def predict(self, input):
        features = input.strip().split()
        prob = self.calprob(features)
        prob.sort(reverse=True)
        return prob


# 调用
model = MaxEnt()
model.load_data('../data/gameLocation.dat')
model.train()
test = 'Dry'
result = model.predict(test)
print test + " predict ->" + str(result)
