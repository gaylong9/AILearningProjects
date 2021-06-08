from numpy.matlib import repmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle as pk
from krk.Class import *

ratioTraining = 0.4
ratioValidation = 0.1
ratioTesting = 0.5


def loadData():
    """从dataset读出数据"""
    data = np.zeros((28056, 6), dtype=float)
    label = np.zeros((28056, 2), dtype=float)
    file = open('dataset/krkopt.data', 'r')
    i = 0
    while True:
        # for i in range(10):
        line = file.readline()
        if line == '':
            break
        data[i] = [ord(line[0:1]) - 96, int(line[2:3]),
                   ord(line[4:5]) - 96, int(line[6:7]),
                   ord(line[8:9]) - 96, int(line[10:11])]
        if line[12:13] == 'd':  # 'draw'
            label[i] = np.array([1, 0])
        else:
            label[i] = np.array([0, 1])
        i = i + 1
    file.close()
    # print(len(data))      # 28056
    # print(data[0])        # [1. 1. 2. 3. 3. 2.]
    # print(label[0])       # [1. 0.]
    return [data, label]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """返回对应的概率值"""
    exp_x = np.exp(x)
    softmax_x = np.zeros(x.shape, dtype=float)
    for i in range(len(x[0])):
        softmax_x[:, i] = exp_x[:, i] / (exp_x[0, i] + exp_x[1, i])

    return softmax_x


def nn_forward(nn, batch_x, batch_y):
    s = len(nn.cost) + 1
    batch_x = batch_x.T
    batch_y = batch_y.T
    m = batch_x.shape[1]    # 样本数量
    nn.a[0] = batch_x

    cost2 = 0
    for k in range(1, nn.depth):
        y = np.dot(nn.W[k - 1], nn.a[k - 1]) + np.tile(nn.b[k - 1], (1, m))  # np.tile就是matlab中的repmat(replicate matrix)

        if nn.batch_normalization:
            nn.E[k - 1] = nn.E[k - 1] * nn.vecNum + np.array([np.sum(y, axis=1)]).T
            nn.S[k - 1] = nn.S[k - 1] ** 2 * (nn.vecNum - 1) + np.array(
                [(m - 1) * np.std(y, ddof=1, axis=1) ** 2]).T  # ddof=1计算无偏估计
            nn.vecNum = nn.vecNum + m
            nn.E[k - 1] = nn.E[k - 1] / nn.vecNum
            nn.S[k - 1] = np.sqrt(nn.S[k - 1] / (nn.vecNum - 1))
            y = (y - np.tile(nn.E[k - 1], (1, m))) / np.tile(nn.S[k - 1] + 0.0001 * np.ones(nn.S[k - 1].shape), (1, m))
            y = nn.Gamma[k - 1] * y + nn.Beta[k - 1]

        if k == nn.depth - 1:
            f = nn.output_function
            if f == 'sigmoid':
                nn.a[k] = sigmoid(y)
            elif f == 'tanh':
                nn.a[k] = np.tanh(y)
            elif f == 'relu':
                nn.a[k] = np.maximum(y, 0)
            elif f == 'softmax':
                nn.a[k] = softmax(y)

        else:
            f = nn.active_function
            if f == 'sigmoid':
                nn.a[k] = sigmoid(y)
            elif f == 'tanh':
                nn.a[k] = np.tanh(y)
            elif f == 'relu':
                nn.a[k] = np.maximum(y, 0)

        cost2 = cost2 + np.sum(nn.W[k - 1] ** 2)

    if nn.encoder == 1:
        roj = np.sum(nn.a[2], axis=1) / m
        nn.cost[s] = 0.5 * np.sum((nn.a[k] - batch_y) ** 2) / m + 0.5 * nn.weight_decay * cost2 + 3 * sum(
            nn.sparsity * np.log(nn.sparsity / roj) + (1 - nn.sparsity) * np.log((1 - nn.sparsity) / (1 - roj)))
    else:
        if nn.objective_function == 'MSE':
            nn.cost[s] = 0.5 / m * sum(sum((nn.a[k] - batch_y) ** 2)) + 0.5 * nn.weight_decay * cost2
        elif nn.objective_function == 'Cross Entropy':
            # 对每个样本计算Y*log(y)，求和得到向量之和，再求和得到一个数字，除以m求平均，后面加上正则项防止过拟合
            nn.cost[s] = -0.5 * sum(sum(batch_y * np.log(nn.a[k]))) / m + 0.5 * nn.weight_decay * cost2
            # nn.cost[s]

    return nn


def nn_backpropagation(nn, batch_y):
    batch_y = batch_y.T
    m = nn.a[0].shape[1]
    nn.theta[1] = 0
    f = nn.output_function
    if f == 'sigmoid':
        nn.theta[nn.depth - 1] = -(batch_y - nn.a[nn.depth - 1]) * nn.a[nn.depth - 1] * (1 - nn.a[nn.depth - 1])
    if f == 'tanh':
        nn.theta[nn.depth - 1] = -(batch_y - nn.a[nn.depth - 1]) * (1 - nn.a[nn.depth - 1] ** 2)
    if f == 'softmax':
        y = np.dot(nn.W[nn.depth - 2], nn.a[nn.depth - 2]) + np.tile(nn.b[nn.depth - 2], (1, m))
        # softmax的最后一个枢纽变量为y-Y
        nn.theta[nn.depth - 1] = nn.a[nn.depth - 1] - batch_y

    if nn.batch_normalization:
        x = np.dot(nn.W[nn.depth - 2], nn.a[nn.depth - 2]) + np.tile(nn.b[nn.depth - 2], (1, m))
        x = (x - np.tile(nn.E[nn.depth - 2], (1, m))) / np.tile(
            nn.S[nn.depth - 2] + 0.0001 * np.ones(nn.S[nn.depth - 2].shape), (1, m))
        temp = nn.theta[nn.depth - 1] * x
        nn.Gamma_grad[nn.depth - 2] = sum(np.mean(temp, axis=1))
        nn.Beta_grad[nn.depth - 2] = sum(np.mean(nn.theta[nn.depth - 1], axis=1))
        nn.theta[nn.depth - 1] = nn.Gamma[nn.depth - 2] * nn.theta[nn.depth - 1] / np.tile(
            (nn.S[nn.depth - 2] + 0.0001), (1, m))

    # 计算梯度，带有正则项
    nn.W_grad[nn.depth - 2] = np.dot(nn.theta[nn.depth - 1], nn.a[nn.depth - 2].T) / m + nn.weight_decay * nn.W[
        nn.depth - 2]
    nn.b_grad[nn.depth - 2] = np.array([np.sum(nn.theta[nn.depth - 1], axis=1) / m]).T
    # 因为np.sum()返回维度为(n,)，会让之后的加法操作错误，所以要转换为(n,1)维度矩阵，下面的也是一样

    f = nn.active_function
    if f == 'sigmoid':
        if nn.encoder == 0:
            for ll in range(1, nn.depth - 1):
                k = nn.depth - ll - 1
                # 此处的层数与视频中的公式层数不一致
                nn.theta[k] = np.dot(nn.W[k].T, nn.theta[k + 1]) * nn.a[k] * (1 - nn.a[k])
                if nn.batch_normalization:
                    x = np.dot(nn.W[k - 1], nn.a[k - 1]) + np.tile(nn.b[k - 1], (1, m))
                    x = (x - np.tile(nn.E[k - 1], (1, m))) / np.tile(nn.S[k - 1] + 0.0001 * np.ones(nn.S[k - 1].shape),
                                                                     (1, m))
                    temp = nn.theta[k] * x
                    nn.Gamma_grad[k - 1] = sum(np.mean(temp, axis=1))
                    nn.Beta_grad[k - 1] = sum(np.mean(nn.theta[k], axis=1))
                    nn.theta[k] = (nn.Gamma[k - 1] * nn.theta[k]) / np.tile((nn.S[k - 1] + 0.0001), (1, m))
                    pass

                # 此处的层数与视频中的公式层数不一致
                nn.W_grad[k - 1] = np.dot(nn.theta[k], nn.a[k - 1].T) / m + nn.weight_decay * nn.W[k - 1]
                nn.b_grad[k - 1] = np.array([np.sum(nn.theta[k], axis=1) / m]).T

        else:
            # encoder完全按照matlab的NN，但貌似是有错误的，用encoder会报错，因为theta[2]（对应matlab的theta{3}）没有赋值
            roj = np.array([np.sum(nn.a[1], axis=1) / m]).T
            temp = (-nn.sparsity / roj + (1 - nn.sparsity) / (1 - roj))
            nn.theta[1] = (np.dot(nn.W[1].T, nn.theta[2]) + nn.beta * repmat(temp, 1, m)) * m
            nn.W_grad[0] = np.dot(nn.theta[1], nn.a[0].T) / m + nn.weight_decay * nn.W[0]
            nn.b_grad[0] = np.array([np.sum(nn.theta[1], axis=1) / m]).T

    elif f == 'tanh':
        for ll in range(1, nn.depth - 1):
            if nn.encoder == 0:
                k = nn.depth - ll - 1
                nn.theta[k] = np.dot(nn.W[k].T, nn.theta[k + 1]) * (1 - nn.a[k] ** 2)
                if nn.batch_normalization:
                    x = np.dot(nn.W[k - 1], nn.a[k - 1]) + np.tile(nn.b[k - 1], (1, m))
                    x = (x - np.tile(nn.E[k - 1], (1, m))) / np.tile(nn.S[k - 1] + 0.0001 * np.ones(nn.S[k - 1].shape),
                                                                     (1, m))
                    temp = nn.theta[k] * x
                    nn.Gamma_grad[k - 1] = sum(np.mean(temp, axis=1))
                    nn.Beta_grad[k - 1] = sum(np.mean(nn.theta[k], axis=1))
                    nn.theta[k] = (nn.Gamma[k - 1] * nn.theta[k]) / np.tile((nn.S[k - 1] + 0.0001), (1, m))
                    pass

                nn.W_grad[k - 1] = np.dot(nn.theta[k], nn.a[k - 1].T) / m + nn.weight_decay * nn.W[k - 1]
                nn.b_grad[k - 1] = np.array([np.sum(nn.theta[k], axis=1) / m]).T

            else:
                roj = np.array([np.sum(nn.a[1], axis=1) / m]).T
                temp = (-nn.sparsity / roj + (1 - nn.sparsity) / (1 - roj))
                nn.theta[1] = (np.dot(nn.W[1].T, nn.theta[2]) + nn.beta * repmat(temp, 1, m)) * m
                nn.W_grad[0] = np.dot(nn.theta[1], nn.a[0].T) / m + nn.weight_decay * nn.W[0]
                nn.b_grad[0] = np.array([np.sum(nn.theta[1], axis=1) / m]).T

    elif f == 'relu':
        if nn.encoder == 0:
            for ll in range(1, nn.depth - 1):
                k = nn.depth - ll - 1
                nn.theta[k] = np.dot(nn.W[k].T, nn.theta[k + 1]) * (nn.a[k] > 0)
                if nn.batch_normalization:
                    x = np.dot(nn.W[k - 1], nn.a[k - 1]) + np.tile(nn.b[k - 1], (1, m))
                    x = (x - np.tile(nn.E[k - 1], (1, m))) / np.tile(nn.S[k - 1] + 0.0001 * np.ones(nn.S[k - 1].shape),
                                                                     (1, m))
                    temp = nn.theta[k] * x
                    nn.Gamma_grad[k - 1] = sum(np.mean(temp, axis=1))
                    nn.Beta_grad[k - 1] = sum(np.mean(nn.theta[k], axis=1))
                    nn.theta[k] = (nn.Gamma[k - 1] * nn.theta[k]) / np.tile((nn.S[k - 1] + 0.0001), (1, m))
                    pass

                nn.W_grad[k - 1] = np.dot(nn.theta[k], nn.a[k - 1].T) / m + nn.weight_decay * nn.W[k - 1]
                nn.b_grad[k - 1] = np.array([np.sum(nn.theta[k], axis=1) / m]).T

        else:
            roj = np.array([np.sum(nn.a[1], axis=1) / m]).T
            temp = (-nn.sparsity / roj + (1 - nn.sparsity) / (1 - roj))
            M = np.maximum(nn.a[1], 0)
            M = M / np.maximum(M, 0.001)

            nn.theta[1] = (np.dot(nn.W[1].T, nn.theta[2]) + nn.beta * repmat(temp, 1, m)) * M
            nn.W_grad[0] = np.dot(nn.theta[1], nn.a[0].T) / m + nn.weight_decay * nn.W[0]
            nn.b_grad[0] = np.array([np.sum(nn.theta[1], axis=1) / m]).T
    return nn


def nn_applygradient(nn):
    method = nn.optimization_method
    if method == 'AdaGrad' or method == 'RMSProp' or method == 'Adam':
        grad_squared = 0
        if nn.batch_normalization == 0:
            for k in range(nn.depth - 1):
                grad_squared = grad_squared + sum(sum(nn.W_grad[k] ** 2)) + sum(nn.b_grad[k] ** 2)
        else:
            for k in range(nn.depth - 1):
                grad_squared = grad_squared + sum(sum(nn.W_grad[k] ** 2)) + sum(nn.b_grad[k] ** 2) + nn.Gamma[k] ** 2 + \
                               nn.Beta[k] ** 2

    for k in range(nn.depth - 1):
        if nn.batch_normalization == 0:
            if method == 'normal':
                nn.W[k] = nn.W[k] - nn.learning_rate * nn.W_grad[k]
                nn.b[k] = nn.b[k] - nn.learning_rate * nn.b_grad[k]

            elif method == 'AdaGrad':
                nn.rW[k] = nn.rW[k] + nn.W_grad[k] ** 2
                nn.rb[k] = nn.rb[k] + nn.b_grad[k] ** 2
                nn.W[k] = nn.W[k] - nn.learning_rate * nn.W_grad[k] / (np.sqrt(nn.rW[k]) + 0.001)
                nn.b[k] = nn.b[k] - nn.learning_rate * nn.b_grad[k] / (np.sqrt(nn.rb[k]) + 0.001)

            elif method == 'Momentum':
                rho = 0.1  # rho = 0.1
                nn.vW[k] = rho * nn.vW[k] - nn.learning_rate * nn.W_grad[k]
                nn.vb[k] = rho * nn.vb[k] - nn.learning_rate * nn.b_grad[k]
                nn.W[k] = nn.W[k] + nn.vW[k]
                nn.b[k] = nn.b[k] + nn.vb[k]

            elif method == 'RMSProp':
                rho = 0.9  # rho = 0.9
                nn.rW[k] = rho * nn.rW[k] + 0.1 * nn.W_grad[k] ** 2
                nn.rb[k] = rho * nn.rb[k] + 0.1 * nn.b_grad[k] ** 2

                nn.W[k] = nn.W[k] - nn.learning_rate * nn.W_grad[k] / (np.sqrt(nn.rW[k]) + 0.001)
                nn.b[k] = nn.b[k] - nn.learning_rate * nn.b_grad[k] / (np.sqrt(nn.rb[k]) + 0.001)  # rho = 0.9

            elif method == 'Adam':
                rho1 = 0.9
                rho2 = 0.999
                nn.sW[k] = 0.9 * nn.sW[k] + 0.1 * nn.W_grad[k]
                nn.sb[k] = 0.9 * nn.sb[k] + 0.1 * nn.b_grad[k]
                nn.rW[k] = 0.999 * nn.rW[k] + 0.001 * nn.W_grad[k] ** 2
                nn.rb[k] = 0.999 * nn.rb[k] + 0.001 * nn.b_grad[k] ** 2

                newS = nn.sW[k] / (1 - rho1)
                newR = nn.rW[k] / (1 - rho2)
                nn.W[k] = nn.W[k] - nn.learning_rate * newS / np.sqrt(newR + 0.00001)
                newS = nn.sb[k] / (1 - rho1)
                newR = nn.rb[k] / (1 - rho2)
                nn.b[k] = nn.b[k] - nn.learning_rate * newS / np.sqrt(
                    newR + 0.00001)  # rho1 = 0.9, rho2 = 0.999, delta = 0.00001

        else:
            if method == 'normal':
                nn.W[k] = nn.W[k] - nn.learning_rate * nn.W_grad[k]
                nn.b[k] = nn.b[k] - nn.learning_rate * nn.b_grad[k]
                nn.Gamma[k] = nn.Gamma[k] - nn.learning_rate * nn.Gamma_grad[k]
                nn.Beta[k] = nn.Beta[k] - nn.learning_rate * nn.Beta_grad[k]

            elif method == 'AdaGrad':
                nn.rW[k] = nn.rW[k] + nn.W_grad[k] ** 2
                nn.rb[k] = nn.rb[k] + nn.b_grad[k] ** 2
                nn.rGamma[k] = nn.rGamma[k] + nn.Gamma_grad[k] ** 2
                nn.rBeta[k] = nn.rBeta[k] + nn.Beta_grad[k] ** 2
                nn.W[k] = nn.W[k] - nn.learning_rate * nn.W_grad[k] / (np.sqrt(nn.rW[k]) + 0.001)
                nn.b[k] = nn.b[k] - nn.learning_rate * nn.b_grad[k] / (np.sqrt(nn.rb[k]) + 0.001)
                nn.Gamma[k] = nn.Gamma[k] - nn.learning_rate * nn.Gamma_grad[k] / (np.sqrt(nn.rGamma[k]) + 0.001)
                nn.Beta[k] = nn.Beta[k] - nn.learning_rate * nn.Beta_grad[k] / (np.sqrt(nn.rBeta[k]) + 0.001)

            elif method == 'RMSProp':
                nn.rW[k] = 0.9 * nn.rW[k] + 0.1 * nn.W_grad[k] ** 2
                nn.rb[k] = 0.9 * nn.rb[k] + 0.1 * nn.b_grad[k] ** 2
                nn.rGamma[k] = 0.9 * nn.rGamma[k] + 0.1 * nn.Gamma_grad[k] ** 2
                nn.rBeta[k] = 0.9 * nn.rBeta[k] + 0.1 * nn.Beta_grad[k] ** 2
                nn.W[k] = nn.W[k] - nn.learning_rate * nn.W_grad[k] / (np.sqrt(nn.rW[k]) + 0.001)
                nn.b[k] = nn.b[k] - nn.learning_rate * nn.b_grad[k] / (np.sqrt(nn.rb[k]) + 0.001)
                nn.Gamma[k] = nn.Gamma[k] - nn.learning_rate * nn.Gamma_grad[k] / (np.sqrt(nn.rGamma[k]) + 0.001)
                nn.Beta[k] = nn.Beta[k] - nn.learning_rate * nn.Beta_grad[k] / (
                            np.sqrt(nn.rBeta[k]) + 0.001)  # rho = 0.9

            elif method == 'Momentum':
                rho = 0.1  # rho = 0.1
                nn.vW[k] = rho * nn.vW[k] - nn.learning_rate * nn.W_grad[k]
                nn.vb[k] = rho * nn.vb[k] - nn.learning_rate * nn.b_grad[k]
                nn.vGamma[k] = rho * nn.vGamma[k] - nn.learning_rate * nn.Gamma_grad[k]
                nn.vBeta[k] = rho * nn.vBeta[k] - nn.learning_rate * nn.Beta_grad[k]
                nn.W[k] = nn.W[k] + nn.vW[k]
                nn.b[k] = nn.b[k] + nn.vb[k]
                nn.Gamma[k] = nn.Gamma[k] + nn.vGamma[k]
                nn.Beta[k] = nn.Beta[k] + nn.vBeta[k]

            elif method == 'Adam':
                nn.sW[k] = 0.9 * nn.sW[k] + 0.1 * nn.W_grad[k]
                nn.sb[k] = 0.9 * nn.sb[k] + 0.1 * nn.b_grad[k]
                nn.sGamma[k] = 0.9 * nn.sGamma[k] + 0.1 * nn.Gamma_grad[k]
                nn.sBeta[k] = 0.9 * nn.sBeta[k] + 0.1 * nn.Beta_grad[k]
                nn.rW[k] = 0.999 * nn.rW[k] + 0.001 * nn.W_grad[k] ** 2
                nn.rb[k] = 0.999 * nn.rb[k] + 0.001 * nn.b_grad[k] ** 2
                nn.rBeta[k] = 0.999 * nn.rBeta[k] + 0.001 * nn.Beta_grad[k] ** 2
                nn.rGamma[k] = 0.999 * nn.rGamma[k] + 0.001 * nn.Gamma_grad[k] ** 2
                nn.W[k] = nn.W[k] - 10 * nn.learning_rate * nn.sW[k] / np.sqrt(1000 * nn.rW[k] + 0.00001)
                nn.b[k] = nn.b[k] - 10 * nn.learning_rate * nn.sb[k] / np.sqrt(1000 * nn.rb[k] + 0.00001)
                nn.Gamma[k] = nn.Gamma[k] - 10 * nn.learning_rate * nn.sGamma[k] / np.sqrt(
                    1000 * nn.rGamma[k] + 0.00001)
                nn.Beta[k] = nn.Beta[k] - 10 * nn.learning_rate * nn.sBeta[k] / np.sqrt(
                    1000 * nn.rBeta[k] + 0.00001)  # rho1 = 0.9, rho2 = 0.999, delta = 0.00001

    return nn


def nn_train(nn, option, train_x, train_y):
    iteration = option.iteration
    batch_size = option.batch_size
    m = train_x.shape[0]    # 样本数
    num_batches = m / batch_size
    for k in range(iteration):
        kk = np.random.permutation(m)
        for l in range(int(num_batches)):
            # 每次送入一个batch训练
            # 用kk即可实现每轮训练都是随机乱序后的数据，无需对数据本身乱序
            batch_x = train_x[kk[l * batch_size: (l + 1) * batch_size],
                      :]  # (l+1)*batch_size也可以改成max((l+1)*batch_size, len(kk))
            batch_y = train_y[kk[l * batch_size: (l + 1) * batch_size], :]
            nn = nn_forward(nn, batch_x, batch_y)
            nn = nn_backpropagation(nn, batch_y)
            nn = nn_applygradient(nn)

    return nn


def nn_predict(nn, batch_x):
    batch_x = batch_x.T
    m = batch_x.shape[1]
    nn.a[0] = batch_x
    # 逐层前向计算至输出值y
    for k in range(1, nn.depth):
        y = np.dot(nn.W[k - 1], nn.a[k - 1]) + np.tile(nn.b[k - 1], (1, m))
        if nn.batch_normalization:
            y = (y - np.tile(nn.E[k - 1], (1, m))) / np.tile(nn.S[k - 1] + 0.0001 * np.ones(nn.S[k - 1].shape), (1, m))
            y = nn.Gamma[k - 1] * y + nn.Beta[k - 1]

        if k == nn.depth - 1:
            f = nn.output_function
            if f == 'sigmoid':
                nn.a[k] = sigmoid(y)
            elif f == 'tanh':
                nn.a[k] = np.tanh(y)
            elif f == 'relu':
                nn.a[k] = np.maximum(y, 0)
            elif f == 'softmax':
                nn.a[k] = softmax(y)

        else:
            f = nn.active_function
            if f == 'sigmoid':
                nn.a[k] = sigmoid(y)
            elif f == 'tanh':
                nn.a[k] = np.tanh(y)
            elif f == 'relu':
                nn.a[k] = np.maximum(y, 0)

    return nn


def nn_test(nn, test_x, test_y):
    nn = nn_predict(nn, test_x)
    y_output = nn.a[nn.depth - 1]
    y_output = y_output.T
    label = np.argmax(y_output, axis=1)  # 按行找出最大元素所在下标
    expectation = np.argmax(test_y, axis=1)
    wrongs = sum(label != expectation)  # 求预测与期望不相等的个数
    success_ratio = 1 - wrongs / test_y.shape[0]

    return wrongs, success_ratio


def testNN():
    data, label = loadData()
    # 随机分配数据集，先将data分为train和非train（以test暂存），再将非train分为test和val
    xTraining, xTesting, yTraining, yTesting = train_test_split(data, label,
                                                                test_size=1 - ratioTraining,
                                                                random_state=0)
    xTesting, xValidation, yTesting, yValidation = train_test_split(xTesting, yTesting,
                                                                    test_size=ratioValidation / ratioTesting,
                                                                    random_state=0)
    # 归一化
    scaler = StandardScaler(copy=False)
    scaler.fit(xTraining)
    scaler.transform(xTraining)
    scaler.transform(xTesting)
    scaler.transform(xValidation)

    nn = NN(layer=[6, 20, 20, 20, 2], active_function='relu', learning_rate=0.01, batch_normalization=1,
            optimization_method='Adam',
            objective_function='Cross Entropy')
    storedNN = None

    option = Option()
    option.batch_size = 50
    option.iteration = 1
    iter = 0
    maxAccuracy = 0
    totalAccuracy = []
    totalCost = []
    maxIteration = 20
    while iter < maxIteration:
        iter = iter + 1
        nn = nn_train(nn, option, xTraining, yTraining)
        totalCost.append(sum(nn.cost.values()) / len(nn.cost.values()))
        # plot(totalCost)
        (wrongs, accuracy) = nn_test(nn, xValidation, yValidation)
        totalAccuracy.append(accuracy)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            storedNN = nn

        print('acc: ', end='')
        print(accuracy)
        print('cost: ', end='')
        print(totalCost[iter - 1])

    file = open('model.pkl', 'wb')
    pk.dump(storedNN, file)
    file.close()
    file = open('xTest.data', 'wb')
    pk.dump(xTesting, file)
    file.close()
    file = open('yTest.data', 'wb')
    pk.dump(yTesting, file)
    file.close()

    plt.subplot(2, 1, 1)
    plt.plot(totalCost, color='blue')
    plt.title('Average Objective Function Value on the Training Set')

    plt.subplot(2, 1, 2)
    plt.plot(totalAccuracy, color='blue')
    plt.ylim([0.8, 1])
    plt.title('Accuracy on the Validation Set')
    plt.tight_layout()
    plt.show()

    wrongs, accuracy = nn_test(storedNN, xTesting, yTesting)
    print('acc on test data: ', accuracy)


def temp():
    file = open('model.pkl', 'rb')
    nn = pk.load(file)
    file.close()

    for i in range(0, nn.depth):
        # print(nn.W[i].shape)
        print(nn.a[i].shape)
        # print(nn.b[i].shape)


testNN()
# temp()