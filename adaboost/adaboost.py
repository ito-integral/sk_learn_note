# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:19:02 2020

@author: 丁敏
"""
import numpy as np
import math

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """dataMatrix 是数据特征矩阵，dimen:是对应的特征列，threshVal:是阈值，
    threshIneq:针对给定阈值的分类方法"""
    retArray =  np.ones((np.shape(dataMatrix)[0], 1)) #预先给定值全部为1的分类列,之后根据分类方法进行修改
    if threshIneq == "lt": #这里的lt指小于等于
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    """dataArr:训练样本数据(行数是样本个数,列数是特征的个数)
    classLabels:训练的样本分类
    D:是指样训练样本数据的概率分布(列向量)
    作用:找最好的单层决策树"""
    dataMatrix = np.mat(dataArr) ; labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix) # m是训练样本的数目, n是特征个数
    bestStamp = {}; bestClasEst = np.mat(np.zeros((m,1)))#初始化最好分类估计
    minError = math.inf
    for i in range(n):
        temp_Arr = dataArr[dataArr[:,i].argsort()] #按矩阵第i+1列从小到大排序
        threval_list = []   
        for j in range(len(temp_Arr) - 1):
            threval = (temp_Arr[:,i][j]+temp_Arr[:,i][j+1]) / 2  
            threval_list.append(threval)   #2分法获得的阈值列表
        for val in threval_list:
            for inequal in ["lt","gt"]:
                predictedVals = stumpClassify(dataMatrix, i, val, inequal) #调用第一个函数
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                print("split: dim %d, thresh %.3f, thresh inequal: %s the weighted error is %.3f " % (i, val, inequal, weightedError))
                if weightedError < minError and weightedError <= 0.5: #错误率要小于50%,这样的分类器才有意义
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStamp['dim'] = i
                    bestStamp['thresh'] = val
                    bestStamp['ineq'] = inequal
    if minError < math.inf:
        return bestStamp, minError, bestClasEst
    else:
        return None
    


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    """基于单层决策树的adaboost算法
    numIt为弱训练器的个数,最后生成一个由numIt个数量的弱训练器列表,并且每个弱分类器有一个对应的alpha"""
    weakClassArr = []
    m = np.shape(dataArr)[0]  #训练数据的个数
    D = np.mat(np.ones((m,1)) / m)  #初始化分布为均匀概率分布
    aggClassEst = np.mat(np.zeros((m,1))) #
    for i in range(numIt):
        if buildStump(dataArr, classLabels, D):
            bestStump, error, classEst = buildStump(dataArr, classLabels, D)
            print("D:",D.T)
            alpha = float(0.5*math.log((1.0-error)/max(error,1e-16)))  #为了确保分母不为0
            bestStump['alpha'] = alpha  #这步其实不必要,只是为了说明一个弱训练器对应一个新的alpha值
            weakClassArr.append(bestStump)
            print("classEst:",classEst.T)
            expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)  #e上的指数
            D = np.multiply(D,np.exp(expon))  #迭代新的概率分布的系数
            D = D/D.sum()  #规范化称为概率分布
        else: break
        aggClassEst += alpha * classEst  #这一步相当于是加法加权弱分类器的分类结果
        print("aggClassEst:",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate,"\n")
        if errorRate == 0.0: break
    return weakClassArr
        
def adaClassify(datToClass,classifierArr):
    """第一个参数是测试数据集,第二个是利用adaBoostTrainDs函数生成的弱分类器列表"""
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0] #测试数据集的个数
    aggClassEst = np.mat(np.zeros((m,1))) #初始化累计分类估计为0
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'], 
                                 classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)
    









    
def main():
    """数据集是周志强西瓜书上 西瓜数据3.0 alpha  第一个特征是密度,第二个特征是含糖率"""
    dataArr = np.array([[0.697,0.460],
                        [0.774,0.376],
                        [0.634,0.264],
                        [0.608,0.318],
                        [0.556,0.215],
                        [0.403,0.237],
                        [0.481,0.149],
                        [0.437,0.211],
                        [0.666,0.091],
                        [0.243,0.267],
                        [0.245,0.057],
                        [0.343,0.099],
                        [0.639,0.161],
                        [0.657,0.198],
                        [0.360,0.370],
                        [0.593,0.042],
                        [0.719,0.103]])
    classLabels = [1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    D = np.mat(np.ones((17,1))/17)   #列向量
    bestStump,minError,bestClasEst = buildStump(dataArr, classLabels, D)
    print(bestStump,minError)
    print(bestClasEst)
    classifierArr = adaBoostTrainDS(dataArr, classLabels, numIt = 11)
    #下面的是例子
    pre_class = adaClassify([0.3,0.5], classifierArr)
    print(pre_class)

    
if __name__ == "__main__":
    main()
    
            
