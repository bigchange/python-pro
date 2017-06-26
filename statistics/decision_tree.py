# coding=utf-8
import math


# 香农熵（信息熵）
def calcShannonEnt(dataSet):
    """
计算训练数据集中的Y随机变量的香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)  # 实例的个数
    labelCounts = {}
    for featVec in dataSet:  # 遍历每个实例，统计标签的频次
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)  # log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征的维度
    :param value: 特征的值
    :return: 符合该特征的所有实例（并且自动移除掉这维特征）
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 删掉这一维特征
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def calcConditionalEntropy(dataSet, i, featList, uniqueVals):
    '''
    计算X_i给定的条件下，Y的条件熵
    :param dataSet:数据集
    :param i:维度i
    :param featList: 数据集特征列表
    :param uniqueVals: 数据集特征集合（该i维特征下取的不同的值）
    :return:条件熵
    '''
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率
        ce += prob * calcShannonEnt(subDataSet)  # ∑ p * H(Y|X=xi) 条件熵的计算
    return ce


def calcInformationGain(dataSet, baseEntropy, i):
    """
    计算信息增益（ID3使用）
    :param dataSet:数据集
    :param baseEntropy:数据集中Y的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益g(dataSet|X_i)
    """
    featList = [example[i] for example in dataSet]  # 第i维特征列表
    uniqueVals = set(featList)  # 转换成集合
    newEntropy = calcConditionalEntropy(dataSet, i, featList, uniqueVals)
    infoGain = baseEntropy - newEntropy  # 信息增益，就是熵的减少，也就是不确定性的减少
    return infoGain


def chooseBestFeatureToSplitByID3(dataSet):
    """
选择最好的数据集划分方式（ID3）
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1  # 最后一列是分类
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有维度特征
        infoGain = calcInformationGain(dataSet, baseEntropy, i)
        if (infoGain > bestInfoGain):  # 选择最大的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最佳特征对应的维度


def chooseBestFeatureToSplitByC45(dataSet):
    """
选择最好的数据集划分方式(C4.5)
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1  # 最后一列是分类
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRate = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有维度特征
        infoGainRate = calcInformationGainRate(dataSet, baseEntropy, i)
        if (infoGainRate > bestInfoGainRate):  # 选择最大的信息增益
            bestInfoGainRate = infoGainRate
            bestFeature = i
    return bestFeature  # 返回最佳特征对应的维度

def calcInformationGainRate(dataSet, baseEntropy, i):
    """
    计算信息增益比
    :param dataSet:数据集
    :param baseEntropy:数据集中Y的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益g(dataSet|X_i)
    """
    return calcInformationGain(dataSet, baseEntropy, i) / baseEntropy