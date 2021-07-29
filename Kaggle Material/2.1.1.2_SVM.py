from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# 使用sklearn自带的手写体数字数据集（只是完整数据集的测试集）
digits = load_digits()
# print(digits.data.shape) # (1797, 64) 1797张图，每张是8*8的像素矩阵
# SVM中，将像素逐行拼接成一个1D向量，损失了空间信息，但是SVM并没有空间信息的学习能力

# 分隔train 和 test
trainX, testX, trainy, testy = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
# print(trainy.shape) # (1347,)
# print(testy.shape) # (450,)
# print(trainX.shape) # (1347, 64)

# 数据标准化
ss = StandardScaler()
trainX = ss.fit_transform(trainX)
testX = ss.transform(testX)

# 线性假设SVM分类器
lsvc = LinearSVC()
lsvc.fit(trainX, trainy)
predicty = lsvc.predict(testX)
print('acc of Linear SVC: %f' % lsvc.score(testX, testy))
print(classification_report(testy, predicty, target_names=digits.target_names.astype(str)))

