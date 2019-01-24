DPCNN详解
=========


# 一. 简介

> ACL2017年中，腾讯AI-lab提出了Deep Pyramid Convolutional Neural Networks for Text Categorization(DPCNN)。论文中提出了一种基于word-level级别的网络-DPCNN，由于上一篇文章介绍的TextCNN 不能通过卷积获得文本的长距离依赖关系，而论文中**DPCNN通过不断加深网络，可以抽取长距离的文本依赖关系**。实验证明在不增加太多计算成本的情况下，增加网络深度就可以获得最佳的准确率。‍



# 二. 结构

> 网络结构如图所示，详细解释如下：

![image](https://github.com/ShaoQiBNU/NLP-DPCNN/blob/master/image/1.png)

## (一) Region embedding——to do 进一步确认

> DPCNN的底层貌似保持了跟TextCNN一样的结构，这里作者将TextCNN的包含多尺寸卷积滤波器的卷积层的卷积结果称之为Region embedding，意思就是对一个文本区域/片段（比如3gram）进行一组卷积操作后生成的embedding。
>
> 对一个3gram进行卷积操作时可以有两种选择，一种是保留词序，也就是设置一组size=3*D的二维卷积核对3gram进行卷积（其中D是word embedding维度）；另一种是不保留词序（即使用词袋模型），首先对3gram中的3个词的embedding取均值得到一个size=D的向量，然后设置一组size=D的一维卷积核对该3gram进行卷积。TextCNN里使用的是保留词序的做法，而DPCNN使用的是词袋模型的做法，DPCNN作者认为前者做法更容易造成过拟合，后者的性能却跟前者差不多
>
> 另外，作者为了进一步提高性能，还使用了**tv-embedding (two-views embedding)进一步提高DPCNN的accuracy**。

## (二) 卷积

### 1. 卷积类型

> 假设输入的序列长度为n，卷积核大小为m，步长(stride)为s，输入序列两端各填补p个零(zero padding)，则该卷积层的输出序列为 (n-m+2p)/s+1。nlp里的序列卷积有以下三类：

#### (1) 窄卷积(narrow convolution)

> 步长 s=1，两端不补零 p=0，卷积后输出长度为 n-m+1。

#### (2) 宽卷积(wide onvolution)

> 步长 s=1，两端补零 p=m-1，卷积后输出长度为 n+m-1。

#### (3) 等长卷积(equal-width convolution)

> 步长 s=1，两端补零 p=(m-1)/2，卷积后输出长度为 n。

### 2. 卷积参数

> 论文里采用的卷积类型是等长卷积，卷积核大小为3，步长 s=1，两端补零 p=(m-1)/2，如图所示：

![image](https://github.com/ShaoQiBNU/NLP-DPCNN/blob/master/image/2.jpg)

## (三) 池化

> 论文里在每一个卷积block之后采用max-pool，size=3，stride=2，序列长度会变成原来的一半，其能够感知到的文本片段就比之前长了一倍。如图所示：

![image](https://github.com/ShaoQiBNU/NLP-DPCNN/blob/master/image/3.jpg)

**注意：图中的max pool的size=2**

## (四) feature maps(filters)的数量

> DPCNN与ResNet很大一个不同就是，在DPCNN中固定了feature map的数量（256），也就是固定住了embedding space的维度，使得网络有可能让整个邻接词（邻接ngram）的合并操作在原始空间或者与原始空间相似的空间中进行。也就是说，整个网络虽然形状上来看是深层的，但是从语义空间上来看完全可以是扁平的。而ResNet则是不断的改变语义空间，使得图像的语义随着网络层的加深也不断的跳向更高level的语义空间。

## (五) Shortcut connections with pre-activation

> 由于在初始化深度CNN时，往往各层权重都是初始化为一个很小的值，这就导致最开始的网络中，后续几乎每层的输入都是接近0，这时网络的输出自然是没意义的，而这些小权重同时也阻碍了梯度的传播，使得网络的初始训练阶段往往要迭代好久才能启动。
>
> 同时，就算网络启动完成，由于深度网络中仿射矩阵（每两层间的连接边）近似连乘，训练过程中网络也非常容易发生梯度爆炸或弥散问题（虽然由于非共享权重，深度CNN网络比RNN网络要好点）。
>
> 上述这两点问题本质就是梯度弥散问题，那么如何解决深度CNN网络的梯度弥散问题呢？当然是膜一下何恺明大神，然后把ResNet的精华拿来用啦～
>
> ResNet中提出的shortcut-connection/skip-connection/residual-connection（残差连接）就是一种非常简单、合理、有效的解决方案。每个block的输入在初始阶段容易是0而无法激活，直接用一条线把region embedding层连接到每个block的输入乃至最终的池化层/输出层。如图所示：

![image](https://github.com/ShaoQiBNU/NLP-DPCNN/blob/master/image/4.png)

> 另外，作者采用的是pre-activation，即先激活再卷积：

![image](https://github.com/ShaoQiBNU/NLP-DPCNN/blob/master/image/5-1.png)
![image](https://github.com/ShaoQiBNU/NLP-DPCNN/blob/master/image/5-2.png)

## (六) 总结

> 由于前面池化层的存在，文本序列的长度会随着block数量的增加呈指数级减少，即
>
> ![num\_blocks=log_2seq\_len](https://www.zhihu.com/equation?tex=num%5C_blocks%3Dlog_2seq%5C_len)
>
> 这导致序列长度随着网络加深呈现金字塔（Pyramid）形状，因此作者将这种深度定制的简化版ResNet称之为Deep “Pyramid” CNN。

![image](https://github.com/ShaoQiBNU/NLP-DPCNN/blob/master/image/6.png)

# 三. 代码

> 采用keras里的文本数据，做二分类，句子长度设置为20，所以short connection只重复了2次，代码如下：

```python
###################### load packages ####################
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Convolution1D, Flatten, Dropout, MaxPool1D, add, ReLU
from keras.utils.np_utils import to_categorical


###################### 参数设置 ####################
######### 只考虑最常见的1000个词 ########
num_words = 1000

######## 句子长度最长设置为20 ########
max_len = 20

######## word dim 词向量维度 ########
word_dim = 300

######## 类别 ########
num_class = 2


###################### load data ####################
def load_data(num_words=1000, max_len=20, num_class=2):
    ######### 导入数据 #########
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

    print(x_train.shape)
    print(x_train[0][:5])

    print(y_train.shape)
    print(y_train[0])

    ###################### preprocess data ####################
    ######## 对文本进行填充，将文本转成相同长度 ########
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

    print(x_train.shape)
    print(x_train[0])

    ######## 对label做one-hot处理 ########
    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)

    print(y_train.shape)
    print(y_train[0])

    return x_train, x_test, y_train, y_test


###################### build network ####################
def DPCNN(num_words=1000, max_len=20, word_dim=300, feature_maps=250, layers=15):

    ######## network structure ########
    #### 输入层 ####
    main_input = Input(shape=(max_len,))

    #### Embedding层 ####
    embedding = Embedding(input_dim=num_words, output_dim=word_dim, input_length=max_len)(main_input)

    ####### 卷积层1 ########
    #### 调整embedding的维度，与feature_maps一致 ####
    block0_input = Convolution1D(feature_maps, 1, padding='same', strides=1, activation='linear')(embedding)

    #### Relu-conv-Relu-conv ####
    block0_relu = ReLU()(embedding)
    block0_conv = Convolution1D(feature_maps, 3, padding='same', strides=1, activation='linear')(block0_relu)

    block0_relu = ReLU()(block0_conv)
    block0_conv = Convolution1D(feature_maps, 3, padding='same', strides=1, activation='linear')(block0_relu)

    #### add ####
    net = add([block0_input, block0_conv])

    ####### short connection ########
    for _ in range(layers):

        #### maxpool 2 ####
        net = MaxPool1D(3, 2)(net)

        #### input ####
        input = net

        #### Relu-conv-Relu-conv ####
        net = ReLU()(net)
        net = Convolution1D(feature_maps, 3, padding='same', strides=1, activation='linear')(net)

        net = ReLU()(net)
        net = Convolution1D(feature_maps, 3, padding='same', strides=1, activation='linear')(net)

        #### add ####
        net = add([input, net])

    #### maxpool ####
    net = MaxPool1D(3, 1)(net)

    #### flatten ####
    flat = Flatten()(net)

    #### drop ####
    drop = Dropout(0.2)(flat)

    #### 输出层 ####
    out = Dense(num_class, activation='softmax')(drop)

    #### 构建模型 ####
    model = Model(inputs=main_input, outputs=out)
    print(model.summary())

    ######## optimization ########
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


###################### main ####################
if __name__ == '__main__':

    ######## load data ########
    x_train, x_test, y_train, y_test = load_data(num_words=1000, max_len=20, num_class=2)

    ######## train and test ########
    model = DPCNN(num_words=1000, max_len=20, word_dim=300, feature_maps=250, layers=2)
    model.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1, validation_data=(x_test, y_test))
```

> 结果如下：

```
(25000,)
[1, 14, 22, 16, 43]
(25000,)
1
(25000, 20)
[ 65  16  38   2  88  12  16 283   5  16   2 113 103  32  15  16   2  19
 178  32]
(25000, 2)
[0. 1.]
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_26 (InputLayer)           (None, 20)           0                                            
__________________________________________________________________________________________________
embedding_26 (Embedding)        (None, 20, 300)      300000      input_26[0][0]                   
__________________________________________________________________________________________________
re_lu_141 (ReLU)                (None, 20, 300)      0           embedding_26[0][0]               
__________________________________________________________________________________________________
conv1d_169 (Conv1D)             (None, 20, 250)      225250      re_lu_141[0][0]                  
__________________________________________________________________________________________________
re_lu_142 (ReLU)                (None, 20, 250)      0           conv1d_169[0][0]                 
__________________________________________________________________________________________________
conv1d_168 (Conv1D)             (None, 20, 250)      75250       embedding_26[0][0]               
__________________________________________________________________________________________________
conv1d_170 (Conv1D)             (None, 20, 250)      187750      re_lu_142[0][0]                  
__________________________________________________________________________________________________
add_71 (Add)                    (None, 20, 250)      0           conv1d_168[0][0]                 
                                                                 conv1d_170[0][0]                 
__________________________________________________________________________________________________
max_pooling1d_76 (MaxPooling1D) (None, 9, 250)       0           add_71[0][0]                     
__________________________________________________________________________________________________
re_lu_143 (ReLU)                (None, 9, 250)       0           max_pooling1d_76[0][0]           
__________________________________________________________________________________________________
conv1d_171 (Conv1D)             (None, 9, 250)       187750      re_lu_143[0][0]                  
__________________________________________________________________________________________________
re_lu_144 (ReLU)                (None, 9, 250)       0           conv1d_171[0][0]                 
__________________________________________________________________________________________________
conv1d_172 (Conv1D)             (None, 9, 250)       187750      re_lu_144[0][0]                  
__________________________________________________________________________________________________
add_72 (Add)                    (None, 9, 250)       0           max_pooling1d_76[0][0]           
                                                                 conv1d_172[0][0]                 
__________________________________________________________________________________________________
max_pooling1d_77 (MaxPooling1D) (None, 4, 250)       0           add_72[0][0]                     
__________________________________________________________________________________________________
re_lu_145 (ReLU)                (None, 4, 250)       0           max_pooling1d_77[0][0]           
__________________________________________________________________________________________________
conv1d_173 (Conv1D)             (None, 4, 250)       187750      re_lu_145[0][0]                  
__________________________________________________________________________________________________
re_lu_146 (ReLU)                (None, 4, 250)       0           conv1d_173[0][0]                 
__________________________________________________________________________________________________
conv1d_174 (Conv1D)             (None, 4, 250)       187750      re_lu_146[0][0]                  
__________________________________________________________________________________________________
add_73 (Add)                    (None, 4, 250)       0           max_pooling1d_77[0][0]           
                                                                 conv1d_174[0][0]                 
__________________________________________________________________________________________________
max_pooling1d_78 (MaxPooling1D) (None, 2, 250)       0           add_73[0][0]                     
__________________________________________________________________________________________________
flatten_15 (Flatten)            (None, 500)          0           max_pooling1d_78[0][0]           
__________________________________________________________________________________________________
dropout_15 (Dropout)            (None, 500)          0           flatten_15[0][0]                 
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 2)            1002        dropout_15[0][0]                 
==================================================================================================
Total params: 1,540,252
Trainable params: 1,540,252
Non-trainable params: 0
__________________________________________________________________________________________________
None
Train on 25000 samples, validate on 25000 samples
Epoch 1/20
25000/25000 [==============================] - 11s 430us/step - loss: 0.6321 - acc: 0.6307 - val_loss: 0.5394 - val_acc: 0.7201
Epoch 2/20
25000/25000 [==============================] - 9s 368us/step - loss: 0.5109 - acc: 0.7406 - val_loss: 0.5107 - val_acc: 0.7376
Epoch 3/20
25000/25000 [==============================] - 9s 364us/step - loss: 0.4596 - acc: 0.7758 - val_loss: 0.5127 - val_acc: 0.7344
Epoch 4/20
25000/25000 [==============================] - 9s 359us/step - loss: 0.4053 - acc: 0.8072 - val_loss: 0.5541 - val_acc: 0.7154
Epoch 5/20
25000/25000 [==============================] - 8s 333us/step - loss: 0.3304 - acc: 0.8524 - val_loss: 0.5786 - val_acc: 0.7373
Epoch 6/20
25000/25000 [==============================] - 8s 314us/step - loss: 0.2368 - acc: 0.9008 - val_loss: 0.6851 - val_acc: 0.7356
Epoch 7/20
25000/25000 [==============================] - 9s 356us/step - loss: 0.1340 - acc: 0.9480 - val_loss: 0.8753 - val_acc: 0.7306
Epoch 8/20
25000/25000 [==============================] - 9s 376us/step - loss: 0.0961 - acc: 0.9636 - val_loss: 0.8967 - val_acc: 0.7298
Epoch 9/20
25000/25000 [==============================] - 9s 370us/step - loss: 0.0578 - acc: 0.9812 - val_loss: 1.2509 - val_acc: 0.7316
Epoch 10/20
25000/25000 [==============================] - 10s 408us/step - loss: 0.0256 - acc: 0.9918 - val_loss: 1.6007 - val_acc: 0.7166
Epoch 11/20
25000/25000 [==============================] - 12s 488us/step - loss: 0.0272 - acc: 0.9902 - val_loss: 1.5382 - val_acc: 0.7283
Epoch 12/20
25000/25000 [==============================] - 10s 389us/step - loss: 0.0280 - acc: 0.9900 - val_loss: 1.5380 - val_acc: 0.7276
Epoch 13/20
25000/25000 [==============================] - 8s 326us/step - loss: 0.0237 - acc: 0.9909 - val_loss: 1.5634 - val_acc: 0.7268
Epoch 14/20
25000/25000 [==============================] - 8s 326us/step - loss: 0.0181 - acc: 0.9940 - val_loss: 1.7204 - val_acc: 0.7277
Epoch 15/20
25000/25000 [==============================] - 8s 333us/step - loss: 0.0158 - acc: 0.9950 - val_loss: 1.7948 - val_acc: 0.7290
Epoch 16/20
25000/25000 [==============================] - 8s 333us/step - loss: 0.0167 - acc: 0.9952 - val_loss: 1.7607 - val_acc: 0.7241
Epoch 17/20
25000/25000 [==============================] - 9s 362us/step - loss: 0.0160 - acc: 0.9945 - val_loss: 1.7799 - val_acc: 0.7272
Epoch 18/20
25000/25000 [==============================] - 10s 409us/step - loss: 0.0187 - acc: 0.9930 - val_loss: 1.8726 - val_acc: 0.7271
Epoch 19/20
25000/25000 [==============================] - 11s 425us/step - loss: 0.0263 - acc: 0.9906 - val_loss: 1.8175 - val_acc: 0.7198
Epoch 20/20
25000/25000 [==============================] - 9s 372us/step - loss: 0.0215 - acc: 0.9919 - val_loss: 1.7274 - val_acc: 0.7263
```

# 参考

https://github.com/hecongqing/TextClassification

https://github.com/Cheneng/DPCNN

https://zhuanlan.zhihu.com/p/35457093

https://mp.weixin.qq.com/s/qsTvdz9Z6hCsIXZXk8RRqA
