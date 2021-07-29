import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 从kaggle下载的数据集，test中没有label
train = pd.read_csv('datasets/titanic/train.csv')
test = pd.read_csv('datasets/titanic/test.csv')
# print(train.head())
# pandas读取的数据都会转为独有的dataframe二维数据表格格式，可用info()查看统计特性
# print(train.info())
# print(test.info())
# 可以看出部分数据确实，数据类型也不同，有数值或字符串，需要预处理
# 选择有效特征
trainX = train[['Pclass', 'Age', 'Sex']]
trainy = train[['Survived']]
# testX = test[['Pclass', 'Age', 'Sex']]
# testy = test[['Survived']]
# print(trainX.info())
# print(trainy.info())

# 补充age的缺失值，用平均数对模型偏离造成的影响小
trainX['Age'].fillna(trainX['Age'].mean(), inplace=True)
# print(trainX.info())

# 将Sex属性由object转为int，DictVector能将object类型的特征按照其不同取值扩展为多种特征，且赋值0/1
vec = DictVectorizer(sparse=False)
trainX = vec.fit_transform(trainX.to_dict(orient='record'))
# print(vec.feature_names_)  # ['Age', 'Pclass', 'Sex=female', 'Sex=male']
# print(trainX[0:5])

# 由于kaggle的test中没有label，所以只好对train集分隔了
trainX, testX, trainy, testy = train_test_split(trainX, trainy, test_size=0.25, random_state=33)
dtc = DecisionTreeClassifier()
dtc.fit(trainX, trainy)
predict = dtc.predict(testX)
print('acc of DecisionTree: %f' % dtc.score(testX, testy))
print(classification_report(testy, predict, target_names=['died', 'survived']))
