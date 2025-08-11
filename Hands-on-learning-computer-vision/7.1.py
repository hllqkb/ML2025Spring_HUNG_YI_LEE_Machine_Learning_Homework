import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_sift_features(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    
    # 检测关键点和描述符
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # 绘制关键点
    img_with_keypoints = cv2.drawKeypoints(gray, keypoints, img)
    
    return keypoints, descriptors, img_with_keypoints

# 使用示例
image_path = 'example.jpg'
keypoints, descriptors, result_img = extract_sift_features(image_path)

# 显示结果
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title(f'Detected {len(keypoints)} SIFT keypoints')
plt.show()