import cv2
from matplotlib import pyplot as plt
# 原图
srcImg = cv2.imread("lenna.jpg",0)
# 5：滤波器的直径（邻域大小
# 100：颜色空间的标准差（控制颜色相似性）
# 100：坐标空间的标准差（控制空间相似性
blurred_img = cv2.bilateralFilter(srcImg, 5, 100, 100)
plt.imshow(blurred_img, cmap='gray')
plt.title("OpenCV Bilateral Filter")
plt.axis('off')
plt.show()