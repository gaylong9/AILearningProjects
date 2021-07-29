from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 联网下载
news = fetch_20newsgroups(subset='all')
# print(len(news.data))  # 18846 条新闻，文本数据，没有特征
# print(news.data[0])

# 分隔
trainX, testX, trainy, testy = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

# 因为新闻数据没有特征，需要先提取出特征向量
vec = CountVectorizer()
trainX = vec.fit_transform(trainX)
testX = vec.transform(testX)

# 朴素贝叶斯模型
mnb = MultinomialNB()
mnb.fit(trainX, trainy)
predicty = mnb.predict(testX)
print('acc of Naive Bayes classifier: %f' % mnb.score(testX, testy))
print(classification_report(testy, predicty, target_names=news.target_names))

