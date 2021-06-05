支持向量机（Support Vector Machine, SVM），监督学习

&nbsp;

# 1. 线性可分

1. 线性可分（Linear Separable）与线性不可分（Nonlinear Separable）：特征空间中，分类能否用直线（二维特征空间）/平面（三维特征空间）/超平面（高维特征空间）完成
2. 线性可分的数学论证（二维空间中）：如有$N$个样本与其标签$\{(X_1,y_1),(X_2,y_2),...,(X_N,y_N)\}$，其中$X_i=[x_{i1}, x_{i2}]^T, y_i=\{+1, -1\}$，若存在$(\omega_1,\omega_2,b)$，有：若$y_i=+1$，则$\omega_1x_{i1}+\omega_2x_{i2}+b>0$，若$y_i=-1$，则$\omega_1x_{i1}+\omega_2x_{i2}+b<0$
3. 向量形式：若有$X_i=[x_{i1} \ x_{i2}]^T \quad \omega=[\omega_1 \ \omega_2]^T$，对于样本集，线性可分为 $y_i(\omega^T X_i+b)>0$，其中的$\omega,b$可看作超平面的法向量和截距，且法向量指向一侧为正类，另一侧为负类
4. 若数据集线性可分，则有无数个超平面也可分

&nbsp;

# 2. 基础定义

既然线性可分时，有无数超平面可分，那么哪个超平面更好呢？

如二分类中，对于一个超平面，将其向两侧平移，至与样本相接。这些与平移后超平面相接触的样本称为**支持向量**，两超平面距离称为**间隔**。**最优的分割超平面，即是间隔最大 且 处于间隔正中间的超平面。**

大间隔的好处：更好的容错能力。

&nbsp;

# 3. 转为最优化问题

已知：样本与其标签$\{(X_i,y_i)\}, i=1\sim N$；

待求：$(\omega, \ b)$；

最小化：$\frac{1}{2}||\omega||^2 = \frac{1}{2}\sum\limits^m_{i=1}\omega_i^2$ （m是$\omega$的维度，即特征空间的维度）；

限制条件：$y_i(\omega^T X_i+b) \geq 1,(i=1\sim N)$ ，即有N个样本，N个限制条件。

## 3.1 推导

点$X_0$到超平面$\omega^Tx+b=0$的距离公式为$d=\frac{|\omega^TX_0+b|}{||\omega||}$；故间隔$r=2\frac{|\omega^Tx_0+b|}{||\omega||}$，其中$x_0$为支持向量，即最终要求
$$
\max_{(\omega,b)} \ 2\frac{|\omega^Tx_0+b|}{||\omega||} \\
限制条件 \ y_i(\omega^T x_i+b)>0
$$
含义为分类正确的情况下，使间隔最大的$\omega,b$，其中$x_i$代表任意向量（数据点）。

另有$\omega^Tx+b=0$与$a\omega^Tx+ab=0(a\neq0)$表示同一超平面，即同一个向量（点）带入这两个超平面式子后，结果不同，同比放缩$a$倍。这说明对于支持向量$x_0$，$|\omega^Tx_0+b|$的值是随着$\omega,b$的变化而变化的，且$\omega,b$的变化不影响其代表的超平面，所以就可以令$|\omega^Tx_0+b|=1$。此时上式变为
$$
\max_{(\omega,b)} \ \frac{2}{||\omega||} = \min_{(\omega,b)}||\omega||  = 
\min_{(\omega,b)}\frac{1}{2}||\omega||^2 
\\
限制条件 \ y_i(\omega^T x_i+b)
= |\omega^T x_i+b| \geq 1
$$
&nbsp;

## 3.2 二次规划问题

属于凸优化问题中的二次规划问题。

二次规划具有以下特点：

1. 目标函数是二次项
2. 限制条件是一次项

凸优化问题要么无解，要么只有唯一解。

&nbsp;

# 4. 线性不可分情况

## 4.1 简单提升

在原N个限制条件下无解，可放松限制条件，变为$y_i(\omega^T x_i+b) \geq 1-\delta_i$，其中$\delta_i$称为松弛变量。

同时松弛变量不应过大，即不能过于宽松，应寻找松弛变量为正，且松弛变量最小的情况，故最终优化问题变为：
$$
\min_{(\omega,b)}(\frac{1}{2}||\omega||^2 + C\sum_{i=1}^{N}\delta_i)
\\
限制条件 \ y_i(\omega^T x_i+b) \geq 1 - \delta_i \ (i=1\sim N) 且
\delta_i \geq 0
$$
其中$C$是为了平衡最小化式子中的二者，人为设置，超参数。此处可设置较高，以迫使松弛变量尽量小。

最小化式子中也可使用$\delta_i^2$。

仍属于凸优化问题。

&nbsp;

## 4.2 核函数

### 4.2.1 背景

当数据情况更复杂时，简单的线性函数分类已无法满足需要。

不同于其他方法常用的直接生成非线性函数，**SVM将特征空间由底维映射到高维，随后在高位空间中继续用线性超平面分类。**

原理：$M$维空间$N$个样本随机打标签正负1，有$M$趋向无穷时，线性可分概率趋向1。直观看待，维度上升，目标数据$\omega,b$的维度上升，算法模型的自由度上升，自然概率更大。

例子：异或，在二维空间中，$x_1=(0,0),x_2=(1,1)$为-1类，$x_3=(1,0),x_4=(0,1)$为1类，线性不可分；当采用$\phi(x)=\phi([a,b]^T)=[a^2,b^2,a,b,ab]^T$映射至五维空间后，则存在$\omega=[-1,-1,-1,-1,6]^T,b=1$解可将升维后的数据分开。

引入映射函数后，最优化问题变为：
$$
\min_{(\omega,b)}(\frac{1}{2}||\omega||^2 + C\sum_{i=1}^{N}\delta_i)
\\
限制条件 \ y_i[\omega^T \phi(x_i)+b] \geq 1 - \delta_i \ (i=1\sim N) 且
\delta_i \geq 0
$$
自然，式(4)中$\omega$的维度与$\phi(x_i)$的维度相同。

&nbsp;

### 4.2.2 核函数

$\phi(x)$的具体形式不需确定，若有$K(x_1,x_2)=\phi(x_1)^T\phi(x_2)$，我们仍可知道测试样本类别信息。这里的$K(x_1,x_2)$称为核函数，结果是一个实数。

核函数$K$和映射函数$\phi$一一对应的充要条件：

1. $K(x_1,x_2)=K(x_2,x_1)$ （可交换性）
2. $\forall C_i(i=1\sim N),\forall N$有$\sum_{i=1}^{N}\sum_{j=1}^{N}C_iC_jK(X_i,X_j)\geq0$ （半正定性）

&nbsp;

# 5. 对偶问题

## 5.1 定义

若有原问题：
$$
\min f(\omega) \\
\begin{aligned}
s.t. g_i(\omega) &\le 0 \quad i=1\sim K \\
h_i(\omega) &= 0 \quad i=1\sim M)
\end{aligned}
$$
定义一个拉格朗日函数：
$$
\begin{aligned}
L(\omega,\alpha,\beta) &= f(\omega) + \sum_{i=1}^{K}\alpha_ig_i(\omega)+\sum_{i=1}^{M}\beta_ih_i(\omega) \\
&= f(\omega)+\alpha^Tg(\omega)+\beta^Th(\omega) \\
其中 \alpha &= [\alpha_1,...,\alpha_K]^T \\
\beta &= [\beta_1,...,\beta_M]^T \\
g(\omega) &= [g_1(\omega),...,g_K(\omega)]^T \\
h(\omega) &= [h_1(\omega),...,h_M(\omega)]^T \\
\end{aligned}
$$
则其**对偶问题**为：
$$
\max \ \theta(\alpha,\beta) \\
其中\theta(\alpha,\beta) = \min \ L(\omega, \alpha, \beta) \ in\ 定义域内的所有\omega \\
s.t. \ \alpha_i \ge 0, i=1\sim K
$$
可得到定理：若$\omega^*$是原问题的解，$(\alpha^*,\beta^*)$是对偶问题的解，则有$f(\omega^*)\ge \theta(\alpha^*,\beta^*)$。

证明：（最后一步由原问题与对偶问题的限制条件得到）
$$
\begin{aligned}
\theta(\alpha^*,\beta^*) &= \inf \ L(\omega, \alpha^*,\beta^*) \\
 &\le L(\omega^*, \alpha^*,\beta^*)\\
 &= f(\omega^*)+\alpha^{*T}g(\omega^*)+\beta^{*T}h(\omega^*) \\
 &\le f(\omega^*)
\end{aligned}
$$
即，原问题的解总是大于等于对偶问题的解。将差值$f(\omega^*)-\theta(\alpha^*,\beta^*)$称为对偶差距。由**强对偶定理**（原函数的目标函数是凸函数，限制条件是线性函数，则其对偶差距为零）和以上证明最后一步，可得到**KKT条件**：对于所有$i=1\sim K$，要么$\alpha_i=0$，要么$g_i(\omega^*)=0$。

&nbsp;

## 5.2 SVM转化为对偶问题

限制条件中的松弛变量取相反数，改为$\delta_i\le0 \ (i=1\sim N)$，故最小化式子变为$\min (\frac{1}{2}||\omega||^2-C\sum_{i=1}^{N}\delta_i)$，另一个限制条件变为$y_i[\omega^T \phi(x_i)+b] \geq 1 + \delta_i$。

综上，SVM整理至满足原问题形式：
$$
\min (\frac{1}{2}||\omega||^2-C\sum_{i=1}^{N}\delta_i) 或 (\frac{1}{2}||\omega||^2+C\sum_{i=1}^{N}\delta_i^2) \\
s.t. \ \delta_i\le 0 \\
1+\delta_i-y_i\omega^T\phi(x_i)-y_ib \le 0 \ (i=1\sim N)
$$
至此，求解函数为凸，限制条件为线性，故满足强对偶定理，可通过求解对偶问题来求解原问题。

另需注意，上节原问题一般形式中所求参数用$\omega$表示，本节SVM中所求参数为$(\omega,b,\delta)$，即此三参数的组合对应一般形式中的$\omega$；且限制条件中，无$h(\omega)$，此处两类限制条件共同组成$g(\omega)$。

以下给出对偶问题形式：因$g(\omega)$由两部分组成，故还是引入了$\beta_i$，应与一般形式中的$\beta_i$区分。
$$
\max \theta(\alpha,\beta)=\inf_{（\omega,\delta,b）}\{\frac{1}{2}||\omega||^2-C\sum_{i=1}^{N}\delta_i+\sum_{i=1}^{N}\beta_i\delta_i+\sum_{i=1}^{N}\alpha_i[1+\delta_i-y_i\omega^T\phi(x_i)-y_ib] \} \\ 
s.t. \ \alpha_i\ge0, \ \beta_i\ge0
$$
对偶问题是拉格朗日函数的极值问题，偏导置零，得到：（$\omega$使用向量求导规则，其余两式使用常规自变量求导规则）
$$
\begin{aligned}
\frac{\partial \theta}{\partial \omega}=\omega-\sum_{i=1}^{N}\alpha_iy_i\phi(x_i) = 0 &\rarr \omega=\sum_{i=1}^{N}\alpha_iy_i\phi(x_i) \\
\frac{\partial \theta}{\partial \delta_i}=-C+\alpha_i+\beta_i = 0 &\rarr \alpha_i+\beta_i=C \\
\frac{\partial \theta}{\partial b}=-\sum_{i=1}^{N}\alpha_iy_i = 0 &\rarr \sum_{i=1}^{N}\alpha_iy_i = 0
\end{aligned}
$$
代回上式，得到最终对偶形式：（TODO：此处$\omega$的代入不懂）
$$
\max \theta(\alpha)=\sum_{i=1}^{N}\alpha_i-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}y_iy_j\alpha_i\alpha_j\phi(x_i)^T\phi(x_j) \\
= \sum_{i=1}^{N}\alpha_i-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}y_iy_j\alpha_i\alpha_jK(x_i,x_j) \\
s.t. 0\le\alpha_i\le C \\
\sum_{i=1}^{N}\alpha_iy_i=0
$$
&nbsp;

# 6. 流程

## 6.1 求$\alpha$

如最终对偶形式，确定核函数后，该对偶问题是一个二次规划问题，不需遍历之类，可以用二次规划、最优化方法求解。利用数据$\{(x_i,y_i)\}i=1\sim N,y_i=\pm1$，求解最终对偶形式式子，得到结果$\alpha$。

可利用$\omega=\sum_{i=1}^{N}\alpha_iy_i\phi(x_i)$求出$\omega$，这里需强调，$\phi(x)$和$\omega$不一定有显式表达。不过不需$\omega$，就可算出$\omega^Tx+b$的值。

&nbsp;

## 6.2 求b

$$
\omega=\sum_{j=1}^{N}\alpha_jy_j\phi(x_j) \\
\omega^T\phi(x_i)=\sum_{j=1}^{N}\alpha_jy_j\phi(x_j)^T\phi(x_i) \\
=\sum_{j=1}^{N}\alpha_jy_jK(x_j,x_i)
$$

另由强对偶定理（KKT？）可知，
$$
\alpha_i[1+\delta_i-y_i\omega^T\phi(x_i)-y_ib]=0\\ 
\beta_i\delta_i=0 \rarr (C-\alpha_i)\delta_i=0
$$
故若$\alpha_i\ne0$且$\alpha_i\ne C$，则必有
$$
\delta_i=0 \\
1+\delta_i-y_i\omega^T\phi(x_i)-y_ib=0
$$
故对任一$0<\alpha_i<C$，可算出b：
$$
\begin{aligned}
b&=\frac{1+\delta_i-y_i\omega^T\phi(x_i)}{y_i} \\
&=\frac{1+0-\sum_{j=1}^{N}\alpha_jy_iy_jK(x_j,x_i)}{y_i}\\
&=\frac{1-\sum_{j=1}^{N}\alpha_jy_iy_jK(x_j,x_i)}{y_i}
\end{aligned}
$$
&nbsp;

## 6.3 预测

$$
\begin{aligned}
\omega^T\phi(x)+b 
&=\sum_{j=1}^{N}\alpha_jy_j\phi(x_j)^T\phi(x)+b \\
&=\sum_{j=1}^{N}\alpha_jy_jK(x_j,x)+b
\end{aligned}
$$

根据上式正负号，可将其分类为正负类。

不知道$\phi(x)$，也可利用核函数求出结果，称为核函数技巧。

&nbsp;

# 7. 核函数

## 7.1 Linear 线性内核

$K(x,y)=x^Ty$

不具有实用意义，因使用线性核函数与不使用的结果一致。（无法升维？）

&nbsp;

## 7.2 Ploy 多项式核

$K(x,y)=(x^Ty+1)^d$

复杂度可调节，$d$越大，升维越高。

&nbsp;

## 7.3 Rbf 高斯径向基函数和

$K(x,y)=e^{-\frac{||x-y||^2}{\sigma^2}}$

$\sigma$超参数，其对应的升维函数$\phi(x)$维度无限。高斯函数具有诸多优秀性质，故本核函数使用最多。

&nbsp;

## 7.4 Tanh sigmoid核

$K(x,y)=tanh(\beta x^Ty+b),\ tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$

其对应的升维函数$\phi(x)$维度无限。

&nbsp;

# 8. 兵王问题

用libsvm和[uci的krk数据集](https://archive.ics.uci.edu/ml/datasets/Chess+(King-Rook+vs.+King))完成兵王问题的分类。代码发在了[Github](https://github.com/gaylong9/AILearningProjects)。

## 8.1 介绍

残局下，白方一个王，一个兵，黑方仅一个王。在不告知计算机象棋规则的情况下，令其判断是白胜还是逼和（黑方的所有落子点都会被吃）。

数据集中有28056组数据，正样本（和棋）有2796组，负样本（白胜）有25260组。

&nbsp;

## 8.2 实际问题中的流程

1. 数据预处理：取部分样本用于训练，其余样本用于测试，此处取5000样本训练
2. 训练样本归一化：训练样本上，求出每个维度的均值和方差；在训练和测试样本上同时归一化；$newX=\frac{X-mean(X)}{std(X)}$；训练样本归一化能使输入特征各维度限定在一个固定范围内，减少不同维度的动态范围不同导致的训练误差
3. 设置参数：
	1. -s：SVM的不同形式，兵王问题是典型二分类，`-s 0`默认即可
	2. -t：核函数，`0123`对应上述四种核函数；`4`是自定义核：SVM中是利用核函数算出所有$K(x_i,x_j)$的值，随后代入式子求解，自定义核即手动输入$K(x_i,x_j)$值的矩阵
	3. -c：原问题中参数`C`，可在$2^{-5}\sim 2^{15}$内搜索
	4. -g：gamma值，对应核函数中不同超参数的值，可在$2^{-15}\sim 2^3$范围内搜索。具体核函数形式以及gamma可在libsvm中-t部分查看
	5. -v：交叉验证折数
4. 指定`C`和`Gamma`的搜索范围，随后交叉验证找出最佳的`C`和`Gamma`
5. 用最佳的`C`和`Gamma`，和完整5000个训练样本，训练最终模型
6. 测试

&nbsp;

# 9. 交叉验证

如五折交叉验证，将训练数据分为ABCDE五组，轮流四组训练一组评估，将五次评估正确率平均，得到最终正确率。

能够在数据量一定的前提下，用更多的数据训练和验证。

若折数极大，达到最极端的每次仅一个数据评估，就是留一法。在数据量极少时采用此法。

&nbsp;

# 10. 性能度量

在不知道先验分布的情况下，单纯用正确率判断系统好坏是没有意义的。

如罕见疾病发病率极低，系统即使全部判为不患病，也能拥有极高的正确率。

引入混淆矩阵概念：

* True Positive（TP）：正样本预测为正的数量或概率
* False Negative（FN）：正样本预测为负的数量或概率
* False Positive（FP）：负样本预测为正的数量或概率
* True Negative（TN）：负样本预测为负的数量或概率

正确率 $ACC=\frac{TP+TN}{TP+FN+FP+TN}$

概率形式下，是以实际正负做归一化，使$TP+FN=1,FP+TN=1$。另有重要性质：若$TP$增加，则$FP$也会增加。故有$FN\darr \iff TP\uarr \iff FP\uarr \iff TN\darr$。

ROC曲线：$FP$为横坐标，$TP$为纵坐标。

绘制ROC：对每个测试样本计算判别式$\omega^T\phi(x)+b$的值，并升序排列；把每个值当作分类时的阈值（判别值>阈值则为正样本），计算此时的TP和FP，保存下来，连成曲线。

越靠近左上的曲线，性能越好。故有以下性能标准：

* AUC（Area Under Curve）：曲线与横轴包围面积，越大越好，代表一个概率值，当随机挑选一个正样本以及一个负样本，当前的分类算法根据计算得到的Score值将这个正样本排在负样本前面的（判别值更高？）概率就是AUC值
* EER（Equal Error Rate）等错误率：连接`(0,1)`和`(1,0)`两点，交ROC曲线于一点，交点横坐标即为EER，此处$FP=FN$。越低越好

&nbsp;

# 11. 多分类问题

1. 1类对k-1类
	1. 若有K类，则构造K个SVM，SVM1判断类1或非类1、SVM2判断类2或非类2、…、SVMK判断类K或非类K。最终得到${\alpha^{(k)}_{i}}_{i=1\sim N}, b^{(k)},k=1\sim K$，K组$\alpha,b$。类别判断时$k_{max}=argmax \sum_{i=1}^{N}\alpha_i^{(k)}y_iK(X_i,X)+b^{(k)},k=1\sim K$，即找到间隔最大的分类
	2. 问题：样本不平衡，正样本仅一类，负样本K-1类，使模型倾向于负类
2. 1类对另一类
	1. 若3类，构造3个SVM，分别是类1vs类2、类1vs类3、类2vs类3。对于测试样本，分别输入各个SVM，得到3个标签，最后用标签投票
	2. 平票：使用各自分数$\sum_{i=1}^{N}\alpha_iy_iK(X_i,X)+b$，如SVM1（类1vs类2）得分0.5，代表该SVM对类1分数0.5，对类2分数-0.5
	3. 解决了训练样本不平衡问题，但要训练$K(K-1)/2$个模型，耗时
3. 树状分类
	1. 如8分类，只需7个分类器：1234vs5678、12vs34、56vs78、1vs2、3vs4、5vs6、7vs8
	2. 但需要每个分类器区分的两类差别是显著的（如1234vs5678，要求1234和5678两大类有足够的差别）；聚类（决策树）可协助完成大分类

&nbsp;

