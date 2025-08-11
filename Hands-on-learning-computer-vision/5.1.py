import cv2
import numpy as np

# 读取图像并转换为灰度图
img = cv2.imread('lenna.jpg', 0)

# Canny边缘检测
# 参数说明：
# img: 输入图像
# threshold1: 第一个阈值，用于边缘连接
# threshold2: 第二个阈值，用于强边缘检测
# apertureSize: Sobel算子的大小，默认为3
# L2gradient: 是否使用更精确的L2范数计算梯度大小(True)，还是使用L1范数(False)
edges = cv2.Canny(img, threshold1=100, threshold2=200, apertureSize=3, L2gradient=True)

# 显示结果
cv2.imshow('Original', img)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()