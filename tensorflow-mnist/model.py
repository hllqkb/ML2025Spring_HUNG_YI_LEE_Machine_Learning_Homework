from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__() # 直接调用父类的构造函数
        self.conv1 = Conv2D(32, 3, activation='relu') # 卷积层
        self.flatten = Flatten() # 展平层
        self.dense1 = Dense(128, activation='relu') # 全连接层
        self.dense2 = Dense(10, activation='softmax') # 输出层

    def call(self, x): # 正向传播过程
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x