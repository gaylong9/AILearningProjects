from libsvm.svmutil import *
from random import *
from math import *
import numpy as np
import pickle as pk
import time
import datetime

nFold = 5  # n折交叉验证
dataSetSize = 28056  # 总样本数量
trainSamplesSize = 5000  # 训练样本数量
evalSamplesSize = dataSetSize - trainSamplesSize  # 测试样本数量
cScale = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]  # SVM中的C和Gamma两个超参数，list用于搜索
gammaScale = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3]
C = [pow(2, i) for i in cScale]
Gamma = [pow(2, i) for i in gammaScale]
scaleRefinedNum = 10  # C和Gamma两超参数，在第二轮细致搜索时的细化程度


def loadData():
    """从dataset读出数据，整理为list与int的形式"""
    data = []
    file = open('dataset/krkopt.data', 'r')
    while True:
        # for i in range(10):
        line = file.readline()
        if line == '':
            break
        X = [ord(line[0:1]) - 96, int(line[2:3]),
             ord(line[4:5]) - 96, int(line[6:7]),
             ord(line[8:9]) - 96, int(line[10:11])]
        if line[12:13] == 'd':  # 'draw'
            data.append([X, 1])
        else:
            data.append([X, -1])
    file.close()
    # print(len(data))      # 28056
    # print(data[0])        # [[1, 1, 2, 3, 3, 2], 1]
    return data


def dataPreProcessing(data):
    """从样本数据中随机取出部分数据作为训练数据，其余作为测试数据，并进行归一化"""
    shuffle(data)  # 将数据打乱
    # 取出前trainSamplesSize个数据作为训练数据，其余数据作为测试数据
    trainX = np.array([data[i][0] for i in range(trainSamplesSize)], dtype=float)  # shape: 5000 * 6
    trainY = np.array([data[i][1] for i in range(trainSamplesSize)])  # shape: 5000
    testX = np.array([data[i][0] for i in range(trainSamplesSize, dataSetSize)], dtype=float)  # shape: 23056 * 6
    testY = np.array([data[i][1] for i in range(trainSamplesSize, dataSetSize)])  # shape: 23056
    # 归一化
    meanX = np.mean(trainX, axis=0)  # shape: 6
    stdX = np.std(trainX, axis=0)  # shape: 6
    for i in range(len(trainX)):
        trainX[i] = (trainX[i] - meanX) / stdX
    for i in range(len(testX)):
        testX[i] = (testX[i] - meanX) / stdX
    # print(trainX[0])
    # print(testX[0])
    return [trainX, trainY, testX, testY]


def crossVal(trainX, trainY):
    """五折交叉验证，找到最佳C和Gamma"""
    print('粗略搜索C和Gamma:')
    # 粗略搜索C和Gamma
    maxRecognitionRate = 0
    bestCIdx, bestGammaIdx = -1, -1
    for i in range(len(C)):
        for j in range(len(Gamma)):
            cmd = ['-c', C[i], '-g', Gamma[j], '-v', nFold, '-h', 0]
            recognitionRate = svm_train(trainY, trainX, cmd)
            print('\033[1;32m' + 'C(' + str(i+1) + '/' + str(len(C)) + '): ' + str(C[i])
                  + ' Gamma(' + str(j+1) + '/' + str(len(Gamma)) + '): ' + str(Gamma[j]) + '\033[0m')
            if recognitionRate > maxRecognitionRate:
                maxRecognitionRate = recognitionRate
                bestCIdx = i
                bestGammaIdx = j
    print('\033[1;32m' + '粗略搜索参数结果：' + '\033[0m')
    print('\033[1;32m' + 'best C: ' + str(C[bestCIdx]) + ', idx: ' + str(bestCIdx) + '\033[0m')
    print('\033[1;32m' + 'best Gamma: ' + str(Gamma[bestGammaIdx]) + ', idx: ' + str(bestGammaIdx) + '\033[0m')
    print('\033[1;32m' + 'best recognitionRate: ' + str(maxRecognitionRate) + '\033[0m')

    print('细致搜索C和Gamma:')
    newLowCScale = 0.5 * (cScale[max(0, bestCIdx - 1)] + cScale[bestCIdx])
    newHighCScale = 0.5 * (cScale[min(len(cScale), bestCIdx + 1)] + cScale[bestCIdx])
    newCScale = list(np.arange(newLowCScale, newHighCScale, (newHighCScale - newLowCScale) / scaleRefinedNum))
    newC = [pow(2, i) for i in newCScale]
    newLowGammaScale = 0.5 * (gammaScale[max(0, bestGammaIdx - 1)] + gammaScale[bestGammaIdx])
    newHighGammaScale = 0.5 * (gammaScale[min(len(gammaScale), bestGammaIdx + 1)] + gammaScale[bestGammaIdx])
    newGammaScale = list(np.arange(newLowGammaScale, newHighGammaScale,
                                   (newHighGammaScale - newLowGammaScale) / scaleRefinedNum))
    newGamma = [pow(2, i) for i in newGammaScale]
    maxRecognitionRate = 0
    bestCIdx, bestGammaIdx = -1, -1
    for i in range(len(newC)):
        for j in range(len(newGamma)):
            cmd = ['-c', newC[i], '-g', newGamma[j], '-v', nFold, '-h', 0]
            recognitionRate = svm_train(trainY, trainX, cmd)
            print('\033[1;32m' + 'newC(' + str(i+1) + '/' + str(len(newC)) + '): ' + str(newC[i])
                  + ' newGamma(' + str(j+1) + '/' + str(len(newGamma)) + '): ' + str(newGamma[j]) + '\033[0m')
            if recognitionRate > maxRecognitionRate:
                maxRecognitionRate = recognitionRate
                bestCIdx = i
                bestGammaIdx = j
    print('\033[1;32m' + '细致搜索参数结果：' + '\033[0m')
    print('\033[1;32m' + 'best C: ' + str(newC[bestCIdx]) + ', idx: ' + str(bestCIdx) + '\033[0m')
    print('\033[1;32m' + 'best Gamma: ' + str(newGamma[bestGammaIdx]) + ', idx: ' + str(bestGammaIdx) + '\033[0m')
    print('\033[1;32m' + 'best recognitionRate: ' + str(maxRecognitionRate) + '\033[0m')
    return [newC[bestCIdx], newGamma[bestGammaIdx]]


def trainAndSaveModel(test=True):
    # 开始时间
    startTick = time.time()
    startTime = time.strftime('%H:%M:%S', time.localtime(startTick))
    # 从数据集加载数据
    data = loadData()
    # 数据预处理：分类、归一化
    trainX, trainY, testX, testY = dataPreProcessing(data)
    # 交叉验证寻找最佳超参数
    bestC, bestGamma = crossVal(trainX, trainY)
    # 利用最佳超参数和所有训练样本训练最终模型
    cmd = ['-c', bestC, '-g', bestGamma]
    model = svm_train(trainY, trainX, cmd)
    # 保存数据与模型
    file = open('testX.data', 'wb')
    pk.dump(testX, file)
    file.close()
    file = open('testY.data', 'wb')
    pk.dump(testY, file)
    file.close()
    svm_save_model('model.pkl', model)
    # file = open('model.pkl', 'wb')
    # pk.dump(model, file)
    # file.close()
    # 测试
    if test:
        labels, acc, vals = svm_predict(testY, testX, model)
    # 结束时间
    endTick = time.time()
    endTime = time.strftime('%H:%M:%S', time.localtime(endTick))
    print('开始时间 ' + startTime)
    print('结束时间 ' + endTime)
    print('所用时长 ' + str(datetime.timedelta(seconds=int(endTick - startTick))))


def loadModelAndTest():
    """从保存的文件中读取测试数据与模型，测试，保存结果"""
    # file = open('model.pkl', 'rb')
    # model = pk.load(file)
    # file.close()
    model = svm_load_model('model.pkl')
    file = open('testX.data', 'rb')
    testX = pk.load(file)
    file.close()
    file = open('testY.data', 'rb')
    testY = pk.load(file)
    file.close()
    labels, acc, vals = svm_predict(testY, testX, model)
    print(labels)
    print(vals)
    file = open('predictAcc.data', 'wb')
    pk.dump(acc, file)
    file.close()
    file = open('predictLabels.data', 'wb')
    pk.dump(labels, file)
    file.close()
    file = open('predictVals.data', 'wb')
    pk.dump(vals, file)
    file.close()


def temp():
    pass


if __name__ == '__main__':
    """重新训练或加载模型以测试"""
    # trainAndSaveModel(test=True)   # 重新训练并保存测试数据和模型
    loadModelAndTest()    # 读取模型并测试数据
    # temp()
