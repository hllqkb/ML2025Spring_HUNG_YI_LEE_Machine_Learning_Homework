import cv2
import numpy as np

# 读取图像并转为灰度图
image = cv2.imread('lenna.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Harris角点检测
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 膨胀结果以标记角点
dst = cv2.dilate(dst, None)

# 阈值筛选并绘制角点
image[dst > 0.01 * dst.max()] = [0, 0, 255]
while True:
    
    cv2.imshow('Harris Corners', image)
    key=cv2.waitKey(0)
    if key==27:
        break
cv2.destroyAllWindows()
