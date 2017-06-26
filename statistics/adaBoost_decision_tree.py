# coding=utf-8
from numpy import mat
from numpy import ones
import numpy as np


def loadSimpData():
    """
加载简单数据集
    :return:
    """
    datMat = np.matrix([[1., 2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    """
用只有一层的树桩决策树对数据进行分类
    :param dataMatrix: 数据
    :param dimen: 特征的下标
    :param threshVal: 阈值
    :param threshIneq: 大于或小于
    :return: 分类结果
    """
    retArray = ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
构建决策树(一个树桩)
    :param dataArr: 数据特征矩阵
    :param classLabels: 标签向量
    :param D: 训练数据的权重向量
    :return: 最佳决策树,最小的错误率加权和,最优预测结果
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(np.zeros((m, 1)))
    minError = np.inf  # 将错误率之和设为正无穷
    for i in range(n):  # 遍历所有维度
        rangeMin = dataMatrix[:, i].min()  # 该维的最小最大值
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # 遍历这个区间
            for inequal in ['lt', 'gt']:  # 遍历大于和小于
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # 使用参数 i, j, lessThan 调用树桩决策树分类
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0  # 预测正确的样本对应的错误率为0,否则为1
                weightedError = D.T * errArr  # 计算错误率加权和
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is
                # %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:  # 记录最优树桩决策树分类器
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
基于单层决策树的ada训练
    :param dataArr: 样本特征矩阵
    :param classLabels: 样本分类向量
    :param numIt: 迭代次数
    :return: 一系列弱分类器及其权重,样本分类结果
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 将每个样本的权重初始化为均等
    aggClassEst = mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels,
                                                D)  # 构建树桩决策树,这是一个若分类器,只能利用一个维度做决策
        # print "D:",D.T
        alpha = float(
            0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 计算 alpha, 防止发生除零错误
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 保存树桩决策树
        # print "classEst: ",classEst.T
        expon = np.multiply(-1 * alpha * mat(classLabels).T, classEst)  # 每个样本对应的指数,当预测值等于y的时候,
        # 恰好为-alpha,否则为alpha
        D = np.multiply(D, np.exp(expon))  # 计算下一个迭代的D向量
        D = D / D.sum()  # 归一化
        # 计算所有分类器的误差,如果为0则终止训练
        aggClassEst += alpha * classEst
        # print "aggClassEst: ",aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != mat(classLabels).T, ones((m,
                                                                                  1)))  # aggClassEst每个元素的符号代表分类结果,如果与y不等则表示错误
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return np.sign(aggClassEst)


def evaluate(aggClassEst, classLabels):
    """
计算准确率与召回率
    :param aggClassEst:
    :param classLabels:
    :return: P, R
    """
    TP = 0.
    FP = 0.
    TN = 0.
    FN = 0.
    for i in range(len(classLabels)):
        if classLabels[i] == 1.0:
            if (np.sign(aggClassEst[i]) == classLabels[i]):
                TP += 1.0
            else:
                FP += 1.0
        else:
            if (np.sign(aggClassEst[i]) == classLabels[i]):
                TN += 1.0
            else:
                FN += 1.0

    return TP / (TP + FP), TP / (TP + FN)


def train_test(datArr, labelArr, datArrTest, labelArrTest, num):
    classifierArr, aggClassEst = adaBoostTrainDS(datArr, labelArr, num)
    prTrain = evaluate(aggClassEst, labelArr)
    aggClassEst = adaClassify(datArrTest, classifierArr)
    prTest = evaluate(aggClassEst, labelArrTest)
    return prTrain, prTest


# 调用
datArr, labelArr = loadSimpData()
classifierArr, aggClassEst = adaBoostTrainDS(datArr, labelArr, 30)
prTrain = evaluate(aggClassEst, labelArr)
print adaClassify([0, 0], classifierArr)