from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

iris = load_iris()
# print(iris.data.shape)  # (150, 4)
# print(iris.DESCR)

trainX, testX, trainy, testy = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
ss = StandardScaler()
trainX = ss.fit_transform(trainX)
testX = ss.transform(testX)
knc = KNeighborsClassifier()
knc.fit(trainX, trainy)
predict = knc.predict(testX)
print('acc of K-Nearest Neighbor Classifier: %f' % knc.score(testX, testy))
print(classification_report(testy, predict, target_names=iris.target_names))

