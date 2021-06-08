人工神经网络ANN，属于AI的仿生学派一支，是对大脑的模拟。数理学派则偏向立足于计算机的物理属性和数学知识。



# 1. 神经元

![神经元细胞](NN/神经元细胞.png)

外部刺激由多个树突传入细胞核，经加工后由轴突传至突触。

将其抽象化为一个数学模型（MP模型）：$y=\phi(\sum^{m}_{i=1}\omega_{i}x_{i}+b)$，向量形式为$y=\phi(W^{T}X+b)$。

![MP模型](NN/MP模型.png)

MP模型并不能真实代表神经元的机制，不过在AI领域效果显著。

&nbsp;

# 2. 感知器算法

Perceptron Learning Algorithm

## 2.1 算法介绍

使用样本，通过机器学习方法，自动求出$W$和$b$。

二分类任务中，输入$(X_i,y_i),i=1\sim N,y_i=\pm 1$，找出$(W,b)$，使得
$$
\begin{align}
y_i=1 &\rightarrow W^TX+b>0 \\
y_i=-1 &\rightarrow W^TX+b<0
\end{align}
$$
若某个数据满足上述条件，则称其获得了平衡。可见此问题与之前的SVM问题一样，要在训练数据线性可分的情况下才能找到$(W,b)$让所有样本平衡。

感知器算法给出了另一种方法求出$(W,b)$：

1. 随机选择$(W,b)$
2. 选取一个样本
	1. 若$W^TX+b>0,y=-1$，则$W=W-X,b=b-1$；
	2. 若$W^TX+b<0,y=1$，则$W=W+X,b=b+1$
3. 再取一个样本，回到2
4. 终止条件：直到任一数据都不满足2中的条件，退出

&nbsp;

调整原理：以情况1为例
$$
\begin{align}
W_{new}^TX+b_{new} 
&= (W_{old}-X)^TX+b_{old}-1 \\
&= (W_{old}^TX+b_{old})-(||X||^T+1) \\
&\leq (W_{old}^TX+b_{old})-1
\end{align}
$$
可见经此调整，判别式值减小了，X离平衡状态更近了。

&nbsp;

## 2.2 收敛证明

感知器算法一定可以停下来吗？

只要数据线性可分，就可以停止。

定义样本的增广向量，若$y_i=1$，则$\hat{X_i}=[X_i \ 1]^T$；若$y_i=-1$，则$\hat{X_i}=[-X_i \ -1]^T$。如此可以将任务简化表达为：
$$
寻找\hat{W}=[W \ b]^T，使得对i=1\sim N，有{\hat{W}}^T\hat{X_i}>0
$$
基于增广向量的感知器算法：

![感知器算法](NN/感知器算法.png)

收敛定理：对于N个样本的增广向量，若存在一个权重向量$\hat{W}_{opt}$，使得对于每个样本都有$\hat{W}_{opt}^T\hat{X_i}>0$，则一定可在有限步内找到一个$\hat{W}$使${\hat{W}}^T\hat{X_i}>0$。

其中，存在权重向量使所有样本平衡，与训练样本线性可分是完全等价的。

且有限步找到的权重向量，也不一定是前提条件中的权重向量，因为若线性可分，是有无数个超平面符合要求的。

收敛定理证明：假设$||\hat{W}_{opt}||=1$，$\hat{W}_{(k)}$是第$k$次调整后的权重向量值，可知对于$\hat{W}_{(k)}$，此时要么所有样本平衡，算法收敛；要么存在某个样本$\hat{W}_{(k)}^T\hat{X}_i\leq0$，则有
$$
\begin{align}
& \hat{W}_{(k+1)}=\hat{W}_{(k)}+\hat{X_i} \\ 
& 两侧同时减去a\hat{W}_{(opt)}: \
\hat{W}_{(k+1)} - a\hat{W}_{(opt)}=\hat{W}_{(k)}-a\hat{W}_{(opt)}+\hat{X_i} \\
& 两侧取模: \
||\hat{W}_{(k+1)} - a\hat{W}_{(opt)}||^2 = ||\hat{W}_{(k)}-a\hat{W}_{(opt)}+\hat{X_i}||^2 \\ 
& = ||\hat{W}_{(k)} - a\hat{W}_{(opt)}||^2 + 2\hat{W}_{(k)}^T\hat{X_i} - 2a\hat{W}_{(opt)}\hat{X_i} \\
& 因为\hat{W}_{(k)}^T\hat{X_i}\leq0，故：\\
& ||\hat{W}_{(k+1)} - a\hat{W}_{(opt)}||^2 \leq ||\hat{W}_{(k)} - a\hat{W}_{(opt)}||^2 + ||\hat{X_i}||^2 - 2a\hat{W}_{(opt)}\hat{X_i} \\
& 又因\hat{W}_{(opt)}\hat{X_i}>0，且||\hat{X_i}||^2有限，故可取足够大的a使得\\
& ||\hat{X_i}||^2 - 2a\hat{W}_{(opt)}\hat{X_i} < -1 \\
& 则有||\hat{W}_{(k+1)} - a\hat{W}_{(opt)}||^2 \leq ||\hat{W}_{(k)} - a\hat{W}_{(opt)}||^2 - 1
\end{align}
$$
可见，$\hat{W}$每更新一次，则它距离$a\hat{W}_{(opt)}$至少接近1（$a$是一个足够大的数）。且最多进行$||\hat{W_0}-a\hat{W}_{(opt)}||^2$次即可收敛到$a\hat{W}_{(opt)}$。当然，更常见的是中途就已收敛（所有样本平衡）退出程序。

其中假设步，与SVM中的证明类似，$\hat{W}$和$a\hat{W}$代表同一超平面，故存在一个$a$使得$||\hat{W}_{opt}||=1$。

&nbsp;

## 2.3 意义

SVM找出间隔最大超平面，而感知器算法找到一个超平面即停止，性能不佳。作为1957年提出的算法，其最大的意义是首次提出了一套机器学习的框架，即：输入$X$，经过模型$f(X,\theta)$，得到输出$Y$；其中$\theta$是待学习的参数，$f$是人为指定的函数，机器学习的过程即为使用训练样本$(X,Y)$，求出$\theta$，如此便可利用$f(X,\theta)$去预测其他样本了。

在感知器算法中，$\theta=(W,b), \ f(X,\theta)=sign(W^TX+b)$，若样本$X$是$M$维，则参数$\theta$则是$M+1$维。

对这个框架的一些认识：

1. 训练数据复杂度和预测函数复杂度应该相匹配：图1中函数复杂度过低，无法拟合数据，称为模型欠拟合；图3函数复杂度过高，导致过拟合

	![训练数据与函数匹配程度](NN/训练数据与函数匹配程度.png)

2. 感知器算法中，只储存$(W,b)$，每次送入一个样本进行计算；而SVM要将所有数据送入内存，解全局优化问题；当下因数据复杂度、维度越来越大，前者类型更受欢迎

&nbsp;

# 3. 多层神经网络

1969~1980，因感知器算法被指出不能解决线性不可分问题，且生活中很多问题都是线性不可分的，导致NN研究陷入停滞。80年代初，多层神经网络被提出，初步具有解决非线性可分问题的能力。

下图为一个简单的两层神经网络的例子：

![两层NN的例子](NN/两层NN的例子.png)
$$
\begin{align}
a_1 &= w_{11}x_1 + w_{12}x_2 + b_1 \ 第一个神经元 \\
a_2 &= w_{21}x_1 + w_{22}x_2 + b_2 \ 第二个神经元 \\
z_1 &= \phi(a_1) \ 非线性函数 \\
z_2 &= \phi(a_2) \ 非线性函数 \\
y &= w_1z_1 + w_2z_2 + b_3 \ 第三个神经元 \\
&= w_1\phi(w_{11}x_1 + w_{12}x_2 + b_1) + w_2\phi(w_{21}x_1 + w_{22}x_2 + b_2) + b_3
\end{align}
$$
这个两层网络，有9个待求参数（各个$w$和$b$）。

此外，若**层与层之间**不加入非线性函数，网络会退化为一个神经元的感知器模型状态（线性模型）。

此处的非线性函数，要使用阶跃函数（y=1(x>0时)，y=0(x<0时)），因为**使用阶跃函数，三层神经网络就可模拟任意的非线性函数。**

这个简单网络中，输出层前未加输出函数，只是简单地全连接后输出。

例如下题，如两线为$w_{11}x_1+w_{12}x_2+b_1=0$和$w_{21}x_1+w_{22}x_2+b_2=0$，将平面分为四个区域，其中蓝色所在的两区域内为类1，标记为1；其余两区域为类2，标记为-1。这个划分就可看作一个非线性函数，可用三层神经网络模拟。

![三层网络例题](NN/三层网络例题.png)

![三层网络例题解](NN/三层网络例题解.jpg)

假设点在线右侧时，代入直线方程计算得大于0；在线左侧时，代入计算小于0。

网络第一层中，使用$w_{11}x_1+w_{12}x_2+b_1$和$w_{21}x_1+w_{22}x_2+b_2$以及激活计算点$X$在两条线的何侧，若在线1的右侧则$z_{11}=1$，否则$z_{12}=0$；若在线2的右侧则$z_{12}=1$，否则$z_{22}=0$。

在第二层，使用$\phi(z_{11}+z_{12}-1)$计算点是否同时处于两线右侧（即右侧的空白区域），若处于则结果$z_{21}=1$，否则为0；使用$\phi(-z_{11}-z_{12}+1)$计算是否同时处于两线左侧（即左侧空白区域），若处于则结果$z_{22}=1$，否则为0。

在第三层，使用$-2z_{21}-2z_{22}+1$分类，若位于空白区域，则为-1，否则为1。

依此为例，可知二维空间内任意非线性函数均可拟合，如圆、椭圆等可用多边形近似，多边形边数对应前两次网络里的神经元数量。高维情况类似。

&nbsp;

# 4. 梯度下降算法

当然，实际问题中，是不知道网络结果，而知道数据与标签。我们要假设一个网络结构（层数与各层神经元个数），使用数据估计网络参数。

网络结构设计两准则：

1. 结构复杂度与问题难易程度相关
2. 结构复杂度与数据数量（训练数据复杂度）相关

网络结构的设计多凭经验，使用数据估计参数则更为“科学”。

继续使用这个简单两层网络，以下介绍估计参数的思路。

![两层NN的例子](NN/两层NN的例子.png)

目的：使用数据$(X,Y)$优化参数$(w,b)$，使网络输出$y$尽可能接近样本标签$Y$。

已知$y=w_1\phi(w_{11}x_1 + w_{12}x_2 + b_1) + w_2\phi(w_{21}x_1 + w_{22}x_2 + b_2) + b_3$，因目的为减小输出与标签的差值，故可写出目标函数
$$
\begin{align}
&\min E(w,b) \\
&= E_{(X,Y)}[(Y-y)^2] \\
&= E_{(X,Y)}\{[Y-(w_1z_1 + w_2z_2 + b_3)]^2\} \\
&= E_{(X,Y)}\{[Y-(w_1\phi(w_{11}x_1 + w_{12}x_2 + b_1) + w_2\phi(w_{21}x_1 + w_{22}x_2 + b_2) + b_3)]^2\}
\end{align}
$$
其中$E_{(X,Y)}$表示遍历训练样本及标签的数学期望，可简单看作对所有样本取均值。

由于此处关于$(w,b)$并非凸函数，无法求出唯一全局极值，采用**梯度下降法**（Gradient Descent）求局部极小值：

1. 随机选取初始值$(w^0,b^0)$

2. 使用迭代算法求目标函数的局部极值，第n步（并非第n层）的更新公式如下，其中$\alpha$为学习率，最重要的超参之一
	$$
	\begin{align}
	w^{n+1} &= w^{n}-\alpha\frac{\partial E}{\partial w}|_{w^n,b^n} \\
	b^{n+1} &= b^{n}-\alpha\frac{\partial E}{\partial b}|_{w^n,b^n}
	\end{align}
	$$

放在上面具体的网络中，要求9个参数，则要不断地求9个偏导数，计算量过大，故引入**后向传播算法**（Back Propagation，BP），利用网络的分层结构，使用链式求导法则，**可以用已经算出的一些偏导数去计算其他偏导数**：

假设目标函数为$\min E(w,b)=\frac{1}{2}(y-Y)^2$，则有
$$
\begin{align}
\frac{\partial E}{\partial y} &= y-Y \\
\frac{\partial E}{\partial a_1} &= \frac{\partial E}{\partial y} \frac{\partial y}{\partial a_1} \\
\frac{\partial y}{\partial a_1} &= \frac{\partial y}{\partial z_1} \frac{\partial z_1}{\partial a_1} \\
可得\frac{\partial E}{\partial a_1} &= \frac{\partial E}{\partial y} \frac{\partial y}{\partial z_1} \frac{\partial z_1}{\partial a_1} \\
&= (y-Y)w_1\phi'(a_1) \\
同理\frac{\partial E}{\partial a_2} &= \frac{\partial E}{\partial y} \frac{\partial y}{\partial z_2} \frac{\partial z_2}{\partial a_2} \\
&= (y-Y)w_2\phi'(a_2) \\
\end{align}
$$
得到$\frac{\partial E}{\partial y},\frac{\partial E}{\partial a_1},\frac{\partial E}{\partial a_2}$后（这三个点即为**枢纽点**，通过它们可以方便地求出$w,b$的偏导），即可方便地求出9个偏导数：
$$
\begin{align}
\frac{\partial E}{\partial w_1} &= \frac{\partial E}{\partial y} \frac{\partial y}{\partial w_1} = (y-Y)z_1 \\
\frac{\partial E}{\partial w_2} &= \frac{\partial E}{\partial y} \frac{\partial y}{\partial w_2} = (y-Y)z_2 \\
\frac{\partial E}{\partial b_3} &= \frac{\partial E}{\partial y} \frac{\partial y}{\partial b_3} = (y-Y) \\
\frac{\partial E}{\partial w_{11}} &= \frac{\partial E}{\partial a_1} \frac{\partial a_1}{\partial w_{11}} = (y-Y)w_1\phi'(a_1)x_1 \\
\frac{\partial E}{\partial w_{12}} &= \frac{\partial E}{\partial a_1} \frac{\partial a_1}{\partial w_{12}} = (y-Y)w_1\phi'(a_1)x_2 \\
\frac{\partial E}{\partial b_1} &= \frac{\partial E}{\partial a_1} \frac{\partial a_1}{\partial b_1} = (y-Y)w_1\phi'(a_1) \\
\frac{\partial E}{\partial w_{21}} &= \frac{\partial E}{\partial a_1} \frac{\partial a_1}{\partial w_{21}} = (y-Y)w_2\phi'(a_2)x_1 \\
\frac{\partial E}{\partial w_{22}} &= \frac{\partial E}{\partial a_1} \frac{\partial a_1}{\partial w_{22}} = (y-Y)w_2\phi'(a_2)x_2 \\
\frac{\partial E}{\partial b_2} &= \frac{\partial E}{\partial a_2} \frac{\partial a_2}{\partial b_2} = (y-Y)w_2\phi'(a_2) \\
\end{align}
$$
推广至一般形式：定义向量$a^l_i$表示第$l$层向量的第$i$个分量，网络最后输出为向量，也带有激活层，即$y=z^L$，目标函数为$\min E(w,b)=\frac{1}{2}(y-Y)^2$
$$
\begin{align}
枢纽变量 \ \delta^m_i &= \frac{\partial E}{\partial a^m_i} \\
对带有激活函数的最后一层 \ \delta^L_i &= \frac{\partial E}{\partial a^L_i}=\frac{\partial E}{\partial y_i}\frac{\partial y_i}{\partial a^L_i}=(y_i-Y_i)\phi'(a^L_i) \\
通过m+1层推导m层 \ \delta^m_i &= \frac{\partial E}{\partial a^m_i} = \sum_{j=1}^{N_{m+1}}\frac{\partial E}{\partial a^{m+1}_j}\frac{\partial a^{m+1}_j}{\partial a^m_i} 
= \sum_{j=1}^{N_{m+1}}\delta^{m+1}_j\frac{\partial a^{m+1}_j}{\partial a^m_i} \\
其中 \ \frac{\partial a^{m+1}_j}{\partial a^m_i} &= \frac{\partial a^{m+1}_j}{\partial z^m_i}\frac{\partial z^m_i}{\partial a^m_i} = W_{ji}^{m+1}\phi'(a^m_i) \\
代入上式得 \ \delta^m_i &= [\sum_{j=1}^{N_{m+1}}\delta^{m+1}_jW_{ji}^{m+1}]\phi'(a^m_i)，如此即可从后向前逐层计算枢纽 \\
最后 \ \frac{\partial E}{\partial W^m_{ji}} &= \delta_j^mz_i^{(m-1)} \\
\frac{\partial E}{\partial b^m_{i}} &= \delta_i^m
\end{align}
$$
&nbsp;

后向传播算法总结：对于每个输入样本，使用所有$z,a,y$，计算枢纽变量（因为仅最后一层的枢纽变量可以直接求出，中间层的枢纽变量按照求导法则也需各种链式，故中间的枢纽变量也应和$w,b$的偏导一样，利用其它枢纽来计算），进而计算各个$w,b$的偏导，用于更新$w,b$，步骤如下：

1. 对每一层的各个神经元，随机选取$(w,b)$
2. 前向计算，计算保留每一层输出值，直至最后一层的输出$y$
3. 设置目标函数$E$，后向传播计算每一个神经元的$\frac{\partial E}{\partial w},\frac{\partial E}{\partial b}$
4. 如式26和27，迭代更新$(w,b)$
5. 回到2，直至$|\frac{\partial E}{\partial w}|,|\frac{\partial E}{\partial b}|$很小

几点注意事项：

1. 若一次输入N个样本（即N个向量，batchSize=N），则将每个向量都代入，都经历1-4步，将N个$\Delta w,\Delta b$取平均后更新，再返回第2步
2. 训练集训练；验证集验证收敛性，收敛后退出训练；测试集测试
3. 初始化参数：自编码器方法

&nbsp;

# 5. 实际应用中的几项改进

上述原理中，诸多内容比较理想化，或不利于实际计算，以下介绍几项改进。

## 5.1 激活函数的改进

阶跃函数在0点无法求导，故改用其他激活函数。[参考](https://zhuanlan.zhihu.com/p/172254089)

1. sigmoid函数：$\phi(x)=\frac{1}{1+e^{-x}}$ $\phi'(x)=\phi(x)[1-\phi(x)]$

	![sigmoid函数](NN/sigmoid函数.png)

	* 缺点：左右两侧导数接近0，参数更新缓慢（称为梯度饱和）；输出非原点对称，导致收敛变慢 [参考](https://liam.page/2018/04/17/zero-centered-active-function/)

2. 双曲正切tanh函数：$\phi(x)=\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$ $\phi'(x)=1-[\phi(x)]^2$

	<img src="NN/双曲正切tanh函数.png" alt="双曲正切tanh函数" style="zoom:80%;" />

	* 输出原点对称了，但仍两侧导数接近0，更新缓慢，且计算形式增加复杂

3. ReLU：$\phi(x)=max(0,x)$

	<img src="NN/ReLU.png" alt="ReLU" style="zoom: 67%;" />

	* 计算形式简单，拟合快，但输出非原点对称，且左侧导数直接为0，导致前向计算若出现小于0，则激活后为0，导致反向传播也为0，参数将无法更新；参数初始化不佳，或较大的lr都会造成Dead ReLU

4. Leaky ReLU：$\phi(x)=max(0.01x, x)$，此处的0.01也可随机取出（随机Leaky ReLU）或专门作为需要学习的参数（PReLU）

5. ELU：$\phi(x)=x(当x>0),\alpha(e^x-1)(当x<0)$，超参取值一般为1

	![ELU](NN/ELU.png)

	* 满足输出的分布是零均值的，可以加快训练速度
	* 激活函数是单侧饱和的，可以更好的收敛（单侧饱和更符合直觉，毕竟负数的大小来代表特征的缺失程度并无意义；且左侧饱和带来的负值统一，能够减少噪声）

6. 总结：先尝试ReLU；不佳时尝试Leaky ReLU和ELU，ELU中指数计算较复杂；随机LReLU和PReLU需要更多的训练样本、算力、时间

&nbsp;

## 5.2 目标函数的改进

分类问题在神经网络中，输出不再是一维变量（一个数字，$\pm1$），而是一个K维变量（对应K类，$[1,0,...,0]^T$），称为独热向量one-hot-vector。

Softmax函数：$y_i=\frac{exp(z_i)}{\sum_{j=1}^{K}exp(z_j)}, i=1\sim K$，（实为归一化）假设最后一隐藏层得到向量$z$，则在该层与最终输出中加入Softmax层，全连接+Softmax函数，得到输出$y$。（即输出函数）

基于交叉熵（Cross-entropy）的目标函数：$E(y)=-\sum_{i=1}^{K}Y_ilog(y_i)$，反映了两个概率分布$Y,y$之间的相似程度，有$E(y)\geq 0$，且当$Y$确定时，当且仅当$y=Y$，$E(y)$取最小值；如此便可采用梯度下降求出局部极值，使$y$尽量接近$Y$。

使用Softmax和交叉熵目标函数时，有$\frac{\partial E}{\partial z}=y-Y$。

&nbsp;

## 5.3 随机梯度下降 SGD

若按上述的每一个样本都更新所有参数，会造成训练极慢，且样本的误差传递到整个网络，使收敛极度缓慢。

引入SGD：

1. 每次输入一批样本（batch），求出梯度的平均值，利用均值修改参数，batchSize从50至200不等。
2. 将所有训练数据，利用batchSize分割为不同的batch；将各个batch逐个送入网络更新参数；所有batch（即整体训练样本）遍历一次，称为一个epoch；要训练多个epoch，每个epoch中要先打乱样本次序，使batch每轮都不同，增加随机性

SGD缺点：

1. $(w,b)$的每个分量的绝对值有大有小，导致参数在较大分量的方向上更新更多，最终造成Z字下降，收敛变慢，AdaGrad对此改善
2. 求梯度策略过于随机，每次batch特征不同，梯度方向呈现随机游走的效果，Momentum改善

梯度下降有诸多算法，后续再写。todo

&nbsp;

# 6. 重要trick

1. 训练集上的目标函数的平均值cost会逐渐减小，若有增大情况，就要停止训练，其原因可能为模型过于简单，无法拟合数据；或者是训练已经很好
2. 使用验证集，训练的本质就是在验证集上取得最大识别率，同时保存在验证集上效果最好的模型
3. lr适度调整，若cost一开始就增加，多为lr过大；若cost变化很小，则lr过小
4. 目标函数可以加入正则项 $min E(w,b)=L(w,b)+\frac{\lambda}{2}||w||^2$，$\lambda$权值衰减系数，即优化要使原目标函数减小，也要是$||w||$减小，降低网络复杂度，使输出更“光滑”，防止过大的$w$造成过拟合
5. 加入正则项的目标函数，前向计算和后向传播都要计算正则项：
	1. 前向计算时对于每个batch，将各层`sum(w ** 2)`求和，即为$||w||^2$
	2. 后向传播计算$w$的偏导时，$gradW^m$要加上$\lambda w^m$
6. 数据归一化：在训练集求出均值和方差，然后在训练集、验证集、测试集上都用该值归一化 $newX=\frac{X-mean(X)}{std(X)}$
7. 参数$(w,b)$初始化：如sigmoid和tanh作为激活函数，若初始化不佳，$|w^Tx+b|$过大，导致梯度消失，反向传播时更新缓慢；故要使$|w^Tx+b|$落在0附近，
	1. 初始化从$(-\frac{1}{\sqrt{d}},\frac{1}{\sqrt{d}})$均匀随机取值，$d$为所在层神经元个数（若样本正态分布，均值0，方差1，各维度无关，$(w,b)$是$(-\frac{1}{\sqrt{d}},\frac{1}{\sqrt{d}})$上的均匀分布，则$|w^Tx+b|$是均值0，方差$\frac{1}{3}$的正态分布
8. Batch Normalization：为进一步改善梯度消失，使每层的输出值都在0附近，对每一层都做基于均值和方差的归一化；在每一层的FC全连接层和激活函数层之间加入BN层，$\hat{x}=\frac{\hat{x}-E[\hat{x}]}{\sqrt{Var[\hat{x}]}}$，其后向传播较为复杂，后续补充todo

&nbsp;





