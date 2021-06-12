1. `tensor.size()` 得到其尺寸

2. `len(tensor)` 得到其长度

3. `criterian(output, label)`计算loss时，要求label为LongTensor类型，且是一维向量 size: [batchSize]，对此，若DataSet在`torchvision.datasets`包中有，则直接配置DataLoader即可，无需操作；若没有，要自己实现DataSet类，并在`__init__`指定`label = torch.zeros(28056, dtype=torch.long)`，读取数据时直接修改`label[i]`即可，如二分类问题，0表示第一类，1表示第二类

4. 自己实现DataSet类，要继承DataSet虚拟类，实现以下三个方法，实现内容只是举例

	```python
	def __init__(self, dataPath, train, trainRatio):
	    self.path = dataPath
	    if train:
	        self.data, self.label = self.dataDivide()
	        self.length = len(data)
	    else:
	        self.data = torch.load(self.path + '../testX.data')
	        self.label = torch.load(self.path + '../testY.data')
	        self.length = len(data)
	
	def __getitem__(self, index: int):
	    return [self.data[index], self.label[index]]
	
	def __len__(self):
	    return self.length
	```

	DataSet类的使用是`trainSet = MyDataSet("path", train=True, trainRatio=0.5)` `testSet = MyDataSet("path", train=False, trainRatio=0.5)`，即训练集和测试集分别是一个实例（或再有个验证集），每个实例内有训练/测试样本data和label

5. 若指定device为cuda，则网络要`net.to(device)`，每个batch也要`to(device)`

6. `_, predict = torch.max(output, 1)`配合softmax（crossEntropy），它会求出最大值的下标，1表示按行求，返回最大值和下标，只使用下标即可，故前者用`_`接收

7. 每个epoch的loss和acc：对于每个batch，criterian得到的loss是一个标量tensor，它是对batch内每个数据的loss均值还是求和，是取决的定义criterian时`nn.CrossEntropyLoss(reduction='sum')` reduction的参数，此处取sum；故每个batch后，得到batch内所有样本的loss之和，同时可以用`torch.max`和batch_label求出正确个数；将每个batch的上述两数字相加，一个epoch结束后即可得到训练集整体的loss之和，和正确个数，此时均除以训练样本长度即可得到本轮的平均loss和正确率acc

8. 

