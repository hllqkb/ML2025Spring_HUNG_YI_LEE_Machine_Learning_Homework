import torch.nn as nn
import torch.nn.functional as f
class LeNet(nn.Module):
    def __init__(self): # 初始化方法
        super(LeNet,self).__init__() # 调用父类的初始化方法
        self.conv1 = nn.Conv2d(3,16,5) # 卷积层1，输入通道3，输出通道16，卷积核大小15
        self.pool1 = nn.MaxPool2d(2,2) # 池化层1，池化核大小2，步长2
        self.conv2 = nn.Conv2d(16,32,5) # 卷积层2，输入通道16，输出通道32，卷积核大小5
        self.pool2 = nn.MaxPool2d(2,2) # 池化层2，池化核大小2，步长2
        self.fc1 = nn.Linear(32*5*5,120) # 全连接层1，输入大小32*5*5，输出大小120
        self.fc2 = nn.Linear(120,84) # 全连接层2，输入大小120，输出大小84
        self.fc3 = nn.Linear(84,10) # 全连接层3，输入大小84，输出大小10

    def forward(self,x): # 前向传播方法
        x = f.relu(self.conv1(x)) # 卷积层1，激活函数relu
        x = self.pool1(x) # 池化层1
        x = f.relu(self.conv2(x)) # 卷积层2，激活函数relu
        x = self.pool2(x) # 池化层2
        x = x.view(-1,32*5*5) # 展平层，将多维度的特征图转化为一维
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x # 不需要softmax层，因为交叉熵损失函数会自动处理softmax
# import torch
# input1=torch.rand([32,3,32,32]) # [batch_size, channel, height, width]随机生成的4D张量（Tensor）
# print(input1.shape)
