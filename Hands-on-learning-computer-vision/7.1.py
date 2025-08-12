import cv2
import time
from matplotlib import pyplot as plt
image1=cv2.imread('image1.jpg')
#  计算特征点提取&生成描述时间
start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
#  使用SIFT查找关键点key points和描述符descriptors
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)
end = time.time()
print("特征点提取&生成描述运行时间:%.2f秒"%(end-start))

kp_image1 = cv2.drawKeypoints(image1, kp1, None)
kp_image2 = cv2.drawKeypoints(image2, kp2, None)

plt.figure()
plt.imshow(kp_image1)
plt.savefig('kp_image1.png', dpi = 300)

plt.figure()
plt.imshow(kp_image2)
plt.savefig('kp_image2.png', dpi = 300)