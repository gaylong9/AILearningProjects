import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report

# 特征名列表
columnNames = ['Sample code number',
               'Clump Thickness',
               'Uniformity of Cell Size',
               'Uniformity of Cell Shape',
               'Marginal Adhesion',
               'Single Epithelial Cell Size',
               'Bare Nuclei',
               'Bland Chromatin',
               'Normal Nucleoli',
               'Mitoses',
               'Class']
# 读取数据
# todo: 不加names，只能读出698/699个数据
data = pd.read_csv('datasets/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=columnNames)
# 替换缺失值
data = data.replace(to_replace='?', value=np.nan)
# 丢弃有缺失值的数据
data = data.dropna(how='any')
# 输出data的数据量和维度
print(data.shape)
print(type(data))
# 由于没有提供测试样本，故需手动分隔train和test
trainX, testX, trainy, testy = train_test_split(data[columnNames[1:10]], data[columnNames[10]],
                                                test_size=0.25, random_state=33)
# 查验结果
print(trainy.value_counts())
print(testy.value_counts())
# testy:
# 2    100
# 4     71
# 表示类2（良性）有100条，类4（恶性）有71条

# 标准化数据，每个维度都是方差1均值0，确保预测结果不会被某些维度上过大的特征主导
ss = StandardScaler()
# 计算trainX的均值和方差，并用其标准化trainX
trainX = ss.fit_transform(trainX)
# 用上一步计算的均值和方差标准化testX
testX = ss.transform(testX)

# 两种分类器，train
lr = LogisticRegression()
sgdc = SGDClassifier()
lr.fit(trainX, trainy)
sgdc.fit(trainX, trainy)
# test
lrPredict = lr.predict(testX)
sgdcPredict = sgdc.predict(testX)


# validation
print('acc of logistic classifier: %f' % lr.score(testX, testy))
print(classification_report(testy, lrPredict, target_names=['Benign', 'Malignant']))
print('acc of SGD classifier: %f' % sgdc.score(testX, testy))
print(classification_report(testy, sgdcPredict, target_names=['Benign', 'Malignant']))


