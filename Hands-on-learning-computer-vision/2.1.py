import matplotlib.pyplot as plt
# 一维卷积
class conv_id():
    def __init__(self,a,b):
        # 输入法信号
        self.a = a
        # 卷积核
        self.b = b
        # 输入信号的坐标
        self.ax= [i for i in range(len(a))]
        # 卷积核的坐标
        self.bx = [i for i in range(len(b))]
    def conv(self):
        # 对输入信号a和卷积核b进行数学上的卷积操作
        c = []
        # c[k] = Σ (a[i - j] * b[j]) 就是一维卷积的公式
        for i in range(len(self.a) + len(self.b) - 1):
            s = 0
            for j in range(len(self.b)):
                if i - j >= 0 and i - j < len(self.a):
                    s += self.a[i - j] * self.b[j]
            c.append(s)
        return c
    def plot(self):
            a, b, ax, bx = self.a, self.b, self.ax, self.bx
            c = self.conv()
            cx = [i for i in range(len(c))]
            # 设置画布 1行3列
            plt.figure(figsize=(12, 4))
            # 输入信号
            plt.subplot(1, 3, 1)
            plt.title('Input')
            plt.bar(ax, a, color='lightcoral', width=0.4)
            plt.plot(ax, a, color='red')
            # 卷积核
            plt.subplot(1, 3, 2)
            plt.title('Kernel')
            plt.bar(bx, b, color='lightgreen', width=0.4)
            plt.plot(bx, b, color='green')
            # 输出信号
            plt.subplot(1, 3, 3)
            plt.title('Output')
            plt.bar(cx, c, color='lightseagreen', width=0.4)
            plt.plot(cx, c, color='teal')
            plt.tight_layout()
            plt.show()

# 使用不同卷积核对三角波信号进行卷积
# 观察不同的卷积核的卷积效果
# 定义输入信号和卷积核
a=[0,1,2,3,2,1,0]
# 冲击函数
k=[1,1,1,1,1]
cov= conv_id(a,k)
# 计算卷积并绘图
cov.plot()