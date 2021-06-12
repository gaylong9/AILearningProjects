import datetime
import time

import torch
from torch.utils.data import DataLoader
import scipy.misc
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import optim
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # out 6@28*28
        self.conv2 = nn.Conv2d(6, 16, 5)            # out 16@10*10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)     # 将卷积结果展开成列向量
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def train():
    # 参数设置
    learning_rate = 1e-3
    batch_size = 100
    epoches = 5
    lenet = LeNet()
    # 数据读入
    trans_img = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST('../datasets', train=True, transform=trans_img, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    criterian = nn.CrossEntropyLoss(reduction='sum')  # loss
    optimizer = optim.Adam(lenet.parameters(), lr=learning_rate)  # optimizer
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    lenet.to(device)

    for i in range(epoches):
        running_loss = 0.
        running_acc = 0.
        for (img, label) in trainloader:
            optimizer.zero_grad()  # 求梯度之前对梯度清零以防梯度累加
            # 需要将数据也转为cuda类型，且tensor.to是非本地转换，需要左值
            if device != torch.device('cpu'):
                img = img.to(device)
                label = label.to(device)
            output = lenet(img)
            # print(output.size())
            print(label)
            exit(0)
            loss = criterian(output, label)
            loss.backward()  # loss反传存到相应的变量结构当中
            optimizer.step()  # 使用计算好的梯度对参数进行更新
            running_loss += loss.item()

            _, predict = torch.max(output, 1)
            correct_num = (predict == label).sum()
            running_acc += correct_num.item()

        running_loss /= len(trainset)
        running_acc /= len(trainset)
        print("[%d/%d] Loss: %.5f, Acc: %.2f" % (i + 1, epoches, running_loss, 100 * running_acc))

    return lenet


def test():
    batch_size = 100
    trans_img = transforms.Compose([transforms.ToTensor()])
    testset = MNIST('/datasets', train=False, transform=trans_img, download=True)
    testloader = DataLoader(testset, batch_size, shuffle=False, num_workers=2)
    running_acc = 0.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for (img, label) in testloader:
        if device != torch.device('cpu'):
            img = img.to(device)
            label = label.to(device)
        output = lenet(img)
        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        running_acc += correct_num.item()
    running_acc /= len(testset)
    return running_acc


if __name__ == '__main__':
    # 开始时间
    startTick = time.time()
    startTime = time.strftime('%H:%M:%S', time.localtime(startTick))

    # 训练
    lenet = train()

    # 保存模型
    torch.save(lenet, './lenet.pkl')

    # 结束时间
    endTick = time.time()
    endTime = time.strftime('%H:%M:%S', time.localtime(endTick))
    print('开始时间 ' + startTime)
    print('结束时间 ' + endTime)
    print('所用时长 ' + str(datetime.timedelta(seconds=int(endTick - startTick))))

    lenet = torch.load('./lenet.pkl')
    test_acc = test()
    print("Test Accuracy:Loss: %.2f" % test_acc)
