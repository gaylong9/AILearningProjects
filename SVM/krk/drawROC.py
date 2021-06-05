import pickle as pk
import matplotlib.pyplot as plt

"""
    ROC曲线：
    对每个测试样本计算判别式wx+b的值，并升序排列；
    把每个值当作分类时的阈值（判别值>阈值则为正样本），计算此时的TP和FP;
    保存下来，连成曲线,FP横坐标，TP纵坐标
"""


def loadFile(filename):
    file = open(filename, 'rb')
    obj = pk.load(file)
    file.close()
    return obj


def saveFile(filename, obj):
    file = open(filename, 'wb')
    pk.dump(obj, file)
    file.close()


def getThird(x):
    return x[2]


#  读取predict的结果
testY = loadFile('testY.data')
labels = loadFile('predictLabels.data')
vals = loadFile('predictVals.data')

# 合并与排序
data = list(zip(testY, labels, vals))
data.sort(key=getThird)

# 计算TP和FP
tps = []
fps = []
for i in range(len(data)):          # 每个val作为阈值计算一次TP和FP
    if (i + 1) % 1000 == 0:
        print(str(i+1) + '/23056')
    tp, fp, tn, fn = 0, 0, 0, 0     # 本轮TP FP个数
    threshold = data[i][2]          # 本轮阈值
    for j in range(len(data)):      # 每个样本根据新阈值分类
        if data[j][2] > threshold:  # 分类为正样本
            if data[j][0] == 1:     # TP
                tp = tp + 1
            else:                   # FP
                fp = fp + 1
        else:                       # 分类为负样本
            if data[j][0] == 1:     # FN
                fn = fn + 1
            else:                   # TN
                tn = tn + 1
    tp = tp / (tp + fn)
    fp = fp / (fp + tn)
    tps.append(tp)
    fps.append(fp)
saveFile('TPs.data', tps)
saveFile('FPs.data', fps)
print(tps)
print(fps)

# 绘制ROC
plt.plot(fps, tps)
plt.title('ROC')
plt.xlabel('FP')
plt.ylabel('TP')
plt.show()
