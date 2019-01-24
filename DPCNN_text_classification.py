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