import random
import time

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


class KrkDataSet(data.Dataset):
    def __init__(self, dataPath, train, trainRatio):
        self.trainSize = 0
        self.testSize = 0
        self.path = dataPath
        self.train = train
        self.trainRatio = trainRatio
        if train:
            self.data, self.label = self.dataDivide()
            self.length = len(self.data)
        else:
            self.data = torch.load(self.path + '/../testX.data')
            self.label = torch.load(self.path + '/../testY.data')
            self.length = len(self.data)

    def __getitem__(self, index: int):
        return [self.data[index], self.label[index]]

    def __len__(self):
        return self.length

    def dataDivide(self):
        """把数据划分为训练数据和测试数据，保存至文件"""
        # 从dataset读出数据
        # criterian中用label要求其为LongTensor类型，且是一维向量 size: [batchSize]
        data = torch.zeros((28056, 6))
        label = torch.zeros(28056, dtype=torch.long)
        file = open(self.path, 'r')
        i = 0
        while True:
            line = file.readline()
            if line == '':
                break
            data[i] = torch.as_tensor([ord(line[0:1]) - 96, int(line[2:3]),
                                       ord(line[4:5]) - 96, int(line[6:7]),
                                       ord(line[8:9]) - 96, int(line[10:11])])
            if line[12:13] == 'd':  # 'draw'
                label[i] = 0
            else:
                label[i] = 1
            i = i + 1
        file.close()
        allDataLen = len(data)
        # shuffle
        index = [i for i in range(allDataLen)]
        data = data[index]
        label = label[index]
        # save
        trainSize = int(allDataLen * self.trainRatio)
        trainX = data[0:trainSize]
        testX = data[trainSize:]
        trainY = label[0:trainSize]
        testY = label[trainSize:]
        torch.save(trainX, self.path + '/../trainX.data')
        torch.save(trainY, self.path + '/../trainY.data')
        torch.save(testX, self.path + '/../testX.data')
        torch.save(testY, self.path + '/../testY.data')
        return [trainX, trainY]


def train():
    # 参数设置
    learningRate = 1e-3
    batchSize = 50
    epoches = 10
    ratioTraining = 0.5
    # ratioValidation = 0.1
    ratioTesting = 0.5
    net = Net()
    trainSet = KrkDataSet("../datasets/krk/krkopt.data", train=True, trainRatio=0.5)
    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=2)
    criterian = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=learningRate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)

    for i in range(epoches):
        runningLoss = 0.
        runningAcc = 0.
        for data, label in trainLoader:
            # 梯度清零防止累加
            optimizer.zero_grad()
            # 数据移至cuda
            if device != torch.device('cpu'):
                data = data.to(device)
                label = label.to(device)
            # 前向传播
            output = net(data)  # size: [batchSize, 2]
            # 计算loss，反向传播并更新
            loss = criterian(output, label)     # 标量tensor
            loss.backward()
            optimizer.step()
            # loss更新
            runningLoss += loss.item()
            _, predict = torch.max(output, 1)
            correctNum = (predict == label).sum()
            runningAcc += correctNum.item()

        runningLoss /= trainSet.length
        runningAcc /= trainSet.length
        print("[%d/%d] Loss: %.5f, Acc: %.2f" % (i + 1, epoches, runningLoss, 100 * runningAcc))

    return net


def test():
    batchSize = 100
    testSet = KrkDataSet('../datasets/krk/krkopt.data', train=False, trainRatio=0.5)
    testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False, num_workers=2)
    runningAcc = 0.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    for data, label in testLoader:
        if device != torch.device('cpu'):
            data = data.to(device)
            label = label.to(device)
        output = net(data)
        _, predict = torch.max(output, 1)
        correctNum = (predict == label).sum()
        runningAcc += correctNum.item()
    acc = runningAcc / testSet.length
    print('acc : %.2f%% (%d/%d)' % (acc*100, int(runningAcc), testSet.length))
    return acc


if __name__ == '__main__':
    # start time
    startTick = time.time()
    startTime = time.strftime('%H:%M:%S', time.localtime(startTick))

    # train
    # net = train()

    # save model
    # torch.save(net, './net.pkl')

    # load model
    net = torch.load('net.pkl')

    # test
    testAcc = test()

