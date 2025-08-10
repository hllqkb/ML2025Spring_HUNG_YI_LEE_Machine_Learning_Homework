from matplotlib import pyplot as plt
import numpy as np
import cv2
import seaborn as sns
class conv_2d():
    def __init__(self, a, b):
        # 输入图像
        self.a = a
        # 卷积核
        self.b = b
        # 输入图像的坐标
        self.ax = [i for i in range(self.a.shape[0])]
        self.ay = [i for i in range(self.a.shape[1])]
        # 卷积核的坐标
        self.bx = [i for i in range(self.b.shape[0])]
        self.by = [i for i in range(self.b.shape[1])]

    def conv(self):
        # 对输入图像a和卷积核b进行数学上的卷积操作
        c = cv2.filter2D(self.a, -1, self.b)
        return c

    def plot(self):
        a = self.a
        b = self.b
        c = self.conv()
        plt.figure(figsize=(12, 4))
        # 输入图像
        plt.subplot(1, 3, 1)
        plt.title('Input')
        # 灰度图
        plt.imshow(a, cmap='gray')
        plt.axis('off')
        # 卷积核
        plt.subplot(1, 3, 2)
        plt.title('Kernel')
        sns.heatmap(b, annot=False, cmap='Greens', cbar=False)
        plt.axis('off')
        # 输出图像
        plt.subplot(1, 3, 3)
        plt.title('Output')
        plt.imshow(c, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
# 冲击信号
img=cv2.imread('lenna.jpg',cv2.IMREAD_GRAYSCALE) # 计算机视觉最经典的一张图片Lenna
# 二维冲击卷积核
# 这个卷积核的效果是：
# 输出图像中的每个像素将是输入图像中对应位置像素的精确复制
# 实际上相当于对图像没有做任何改变（单位操作）
# 因为卷积核只在中心有1，其他都是0，相当于只取源图像对应位置的像素值
size=15
# 创建了一个15×15的全零矩阵k1
k1=np.zeros((size,size))
# 因为二维的核函数的大小是n*n的，因此在实现方波信号时我们需要除以(size*size)
k2=np.ones((size,size))/size** 2
# 计算中间位置mid = (15-1)//2 = 7
mid=(size-1)//2
# 在中间位置设置为1
k1[mid,mid]=1
# 计算卷积并绘图
# cov = conv_2d(img, k1)
cov = conv_2d(img, k2)
cov.plot()