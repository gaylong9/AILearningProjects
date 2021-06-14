# 1. 背景

神经网络缺点：

1. 数学不够优美，只能求得局部极值，算法性能与初始值有关；
2. 不可解释
3. 超参很多，调参玄学
4. 样本需求量大

&nbsp;

# 2. 自编码器

Auto-encoder：参数初始化方法，缓解初始化在较差的局部极值附近。

采用分层初始化思想：如要训练N层网络，则先用输入X训练第一层，其输出可以看作对输入自身的编码过程，故称为自编码器；随后固定第一层，训练前两层，BP仅作用于第二层。

![auto-encoder1](Deep Learing/auto-encoder1.png)

![auto-encoder2](Deep Learing/auto-encoder2.png)

![auto-encoder3](Deep Learing/auto-encoder3.png)

&nbsp;

# 3. CNN

卷积神经网络

## 3.1 卷积核

卷积核参数也是待定的，需要BP更新。卷积核可看作一个权值共享层，卷积核的参数就是共享的$w$，且$b=0$，如此可方便地写出偏导。

![卷积核](Deep Learing/卷积核.png)

![卷积式子](Deep Learing/卷积式子.png)

上面的卷积核对应下面的网络。

![卷积对应权值共享网络](Deep Learing/卷积对应权值共享网络.png)

![卷积偏导](Deep Learing/卷积偏导.png)



卷积大小的公式：如图像$M*N$，卷积核$m*n$，步长$(P,Q)$，则输出特征图长为$H=floor(\frac{M-m}{P})+1$，宽为$W=floor(\frac{N-n}{Q})+1$。

&nbsp;

## 3.2 降采样/池化

subsampling（池化 pooling）。

LeNet使用平均降采样，即范围内像素取平均，作为结果。如相邻4个像素平均降采样，则有$Y=(x_1+x_2+x_3+x_4)/4$，故$\frac{\partial E}{\partial x_1}=\frac{\partial E}{\partial x_2}=\frac{\partial E}{\partial x_3}=\frac{\partial E}{\partial x_4}=\frac{1}{4}\frac{\partial E}{\partial y}$。

AlexNet中使用最大池化，在梯度传递时，仅最大处有梯度，每次需要更新的参数减少，收敛加快。

&nbsp;

## 3.3 dropout

避免快速拟合导致过拟合，每次随机丢弃一部分神经元。被丢弃的神经元不参与训练过程，参数也不更新。

&nbsp;

## 3.4 数据扩增 

将原图水平翻转、随机选出局部图像、引入噪声等方法。

&nbsp;

# 4. 经典改进

## 4.1 VGGNet

对AlexNet的改进，增加深度，用多个3$\times$3的卷积核叠加代替大卷积核，增加了感受野。

感受野：特征图的像素点，在输入图片上映射区域的大小。

感受野计算公式：$RF_i=(RF_{i-1}-1)\times stride_{i-1}+KSIZE_i$，其中$RF_i$为第i层感受野，$stride_i$是第i层步长，$KSIZE_i$是第i层卷积核大小。

如下图，使用两个3$\times$3的卷积核，可以替代一个5$\times$5的卷积核（步长均为1），且此时感受野是5$\times$5。

![VGGNet卷积核替代](Deep Learing/VGGNet卷积核替代.png)

这样取代的意义：待求参数从25减少到18。当然，这会带来计算和存储的更大开销。故VGG是一个计算和存储开销较大的网络。

&nbsp;

## 4.2 GoogleNet

对AlexNet的改进。提出Inception结构，将一些1$\times$1、3$\times$3、5$\times$5的小卷积核，以固定方式组合，代替大卷积核。同样达到了增大感受野和减少参数的目的。

&nbsp;

## 4.3 ResNet

越深的网络，虽然理论上能带来更高的准确率，但复杂的网络难以收敛，训练效果不佳。浅网络训练和测试各阶段都有更优表现。故ResNet提出将浅层（20层）的输出直接加入深层（56层）中。同时使用线性变换$X'=W^TX+B$解决两层之间纬度不同的问题，其中$W,B$作为待求变量学习。

这也开辟了当下更注重小而深的网络的趋势。

&nbsp;

## 4.4 L-Softmax

此改进是在人脸识别中讲述。

基础softmax损失函数为
$$
L_i=\frac{1}{N}\sum_i-log(\frac{e^{f_{yi}}}{\sum_je^{f_j}})
$$
其中$N$是样本数量，$i$代表第i个训练样本，$j$代表第j个类别，$f_{yi}=W^T_{yi}x_i$是最后一层全连接层的输出，代表第i个样本所属类别的分数，用角度表示为
$$
L_i=-log(\frac{e^{||W_{yi}||||x_i||cos(\theta_{yi})}}{\sum_je^{||W_j||||x_i}cos(\theta_j)})
$$
Large-margin softmax：

如二分类问题，x属于类1，正确分类要满足$||W_1||||x||cos(\theta_1)>||W_2||||x||cos(\theta_2)$，L-softmax将其替换为$||W_1||||x||cos(m\theta_1)>||W_2||||x||cos(\theta_2),(0\le\theta_1\le\frac{\pi}{m})$，$m$是一个正整数，cos范围内单增，故新式满足时旧式一定满足。这能使模型学到类间距离更大、类内距离更小的特征。

改进后损失函数：
$$
L_i=-log(\frac{e^{||W_{yi}||||x_i||\phi(\theta_{yi})}}{e^{||W_{yi}||||x_i||\phi(\theta_{yi})}+\sum_{j\ne y_i}e^{||W_j||||x_i}cos(\theta_j)}) \\
\phi(\theta)=
\left\{ 
\begin{array}{}
cos(m\theta), \ 0\le\theta\le\frac{\pi}{m} \\
D(\theta),  \ \frac{\pi}{m}<\theta\le\pi
\end{array}
\right.
$$
基于此改进，后续有cosface、arcface等方法。

&nbsp;

# 5. 经典应用

## 5.1 人脸识别

如DeepID，网络倒数第二层是一个160维的向量，最后一层softmax输出维度是类别数。但因数据库中类别数巨大，以万计，测试中这个输出十分冗余，故可以在训练时保留softmax，而测试时去掉，直接使用倒数第二层的160维向量，利用欧氏距离、余弦距离等量度，通过阈值获得结果。

&nbsp;

## 5.2 目标检测与分割

### 5.2.1 定位与识别

分类，还要定位。可在输出最后加四个参数，表示左上角点的坐标和方框长宽，同时图片需要归一化。

### 5.2.2 多目标检测

多个目标的分类与定位。若能找到一个区域，其中只有一个目标，则可退化为上一种情况。

RCNN提出，遍历各种方框不现实，它提出用selective search产生候选方框proposal，将方框输入CNN，最后用SVM判断检测结果是否正确。

1. selective search：用efficient graph-basedimage segmentation算法，将图片过分割，此时各个region很小；将相邻region判断相似度并融合，每个区域对应一个bounding box
2. cnn：将候选框scale and resize（归一化尺寸），输入CNN
3. svm判断该区域到底有没有目标

&nbsp;

但R-CNN运行极其耗时，故提出Fast R-CNN，使用ROI-Pooling加速CNN特征提取过程：观察到候选框多有重叠，逐个CNN卷积，会导致重叠区域的重复计算，故可先用CNN对整幅图卷积，在中间某一层的特征图用ROI-Pooling归一化每个候选框区域的输出，如此一轮卷积即可处理所有候选框。

ROI-Pooling：根据不同候选框在某一层特征图的感受野，成比例地利用maxpooling，将每个候选框经池化后获得的特征图在维度（尺寸）上一致；这些一致的特征通过全连接层计算结果，输出仍是分类的softmax+4个坐标。
&nbsp;

Fast R-CNN仅对CNN步骤改进，在第一步生成proposal时仍需耗费较多时间，故有Faster R-CNN提出。它改用深度学习产生proposal。

在卷积后的特征图上滑动窗口，用不同长宽比的矩形作为proposal，用一个小网络判断该proposal是否存在目标；若存在，则ROI Pooling归一化，输入CNN计算。

&nbsp;

多目标检测的YOLO也是热门网络之一。将输入图像分成S$\times$S个格子，若某个物体的Ground Truth的中心位置的坐标落入到某个格子，则该格子负责检测该物体。每个格子预测B个bounding box及其置信度，以及C个类别概率。bbox的信息是$(X,Y,w,h)$，代表物体中心位置相对格子位置的偏移及宽高，均被归一化。置信度则是是否包含物体以及包含物体时位置的准确性。

YOLOv1最后全连接层，输出维度$S\times S\times (B\times 5+C)$。网络中间结构简洁直接。

&nbsp;

MTCNN检测人脸与眼睛鼻子嘴巴五个特征点。第一段P-Net，检测人脸，产生候选框，用回归向量校准候选框，通过非极大值抑制NMS合并高度重合的候选框；R-Net输出候选框置信度和回归向量，通过置信度削减候选框数量，通过边界框回归和NMS精调候选框位置；O-Net削减数量，精调位置。

&nbsp;

### 5.2.3 语义分隔

全卷积网络，输出是一个与输入尺寸一致的图像，各卷积层特征图先缩小后放大。

上采样层up-sampling用于特征图放大。如maxpooling的上采样，记录下采样时各区域最大像素的位置，在上采样时将小特征图的值对照到生成的大特征图位置，其余位置补0。卷积层的上采样是通过小特征图周围padding（补0）来卷积出大特征图。

全卷积网络不仅可以用于语义分隔，因其输入输出特点，可以用于许多cv领域，如人群计数。

&nbsp;

## 5.3 时间序列

语音识别、行为识别等是针对一段时间内的内容进行判断。

有循环神经网络RNN，LSTM两个经典网络。

RNN：状态$h_t=f_W(h_{t-1},x_t)=tanh(W_{hh}h_{t-1}+W_{xh}x_t)$ 输出$y_t=W_{hy}h_t$，可用于解决多输入多输出、多输入一输出、一输入多输出（文本生成、图像描述）。

传统RNN的状态迭代不够复杂，但若增加网络结构，又导致难以训练。LSTM设计了巧妙且复杂的网络，得到了不错的效果。

&nbsp;

## 5.4 GAN

使用两个网络，一个生成器G，一个判别器D，生成器负责生成作品，判别器负责检验真假。可以把生成器看作训练样本概率分布的学习。训练中二者交替训练。当模型容量足够时，可达到纳什均衡，判别器判别结果均为一半，无法区分。训练结束后，只需输入一个随机噪声，用生成器生成即可。

最近应用：人脸生成、侧脸转正、图像翻译

&nbsp;

