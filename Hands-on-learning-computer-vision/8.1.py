import cv2
import numpy as np

def stitch_images(images, ratio=0.75, reproj_thresh=4.0, show_matches=False):
    """
    图像拼接函数
    
    参数:
        images: 要拼接的图像列表
        ratio: Lowe's ratio test参数
        reproj_thresh: RANSAC重投影阈值
        show_matches: 是否显示特征匹配结果
        
    返回:
        拼接后的图像
    """
    # 初始化OpenCV的SIFT特征检测器
    sift = cv2.SIFT_create()
    
    # 检测关键点和描述符
    (kpsA, featuresA) = sift.detectAndCompute(images[0], None)
    (kpsB, featuresB) = sift.detectAndCompute(images[1], None)
    
    # 匹配特征点
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    raw_matches = matcher.knnMatch(featuresA, featuresB, 2)
    
    # 应用Lowe's ratio test筛选好的匹配点
    good_matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            good_matches.append((m[0].trainIdx, m[0].queryIdx))
    
    # 至少需要4个匹配点才能计算单应性矩阵
    if len(good_matches) > 4:
        ptsA = np.float32([kpsA[i].pt for (_, i) in good_matches])
        ptsB = np.float32([kpsB[i].pt for (i, _) in good_matches])
        
        # 计算单应性矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
        
        # 拼接图像
        result = cv2.warpPerspective(images[0], H, 
                                    (images[0].shape[1] + images[1].shape[1], 
                                     images[0].shape[0]))
        result[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]
        
        # 如果需要显示匹配结果
        if show_matches:
            vis = np.zeros((max(images[0].shape[0], images[1].shape[0]), 
                           images[0].shape[1] + images[1].shape[1], 3), dtype=np.uint8)
            vis[0:images[0].shape[0], 0:images[0].shape[1]] = images[0]
            vis[0:images[1].shape[0], images[0].shape[1]:] = images[1]
            
            for ((trainIdx, queryIdx), s) in zip(good_matches, status):
                if s == 1:
                    ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
                    ptB = (int(kpsB[trainIdx].pt[0]) + images[0].shape[1], 
                           int(kpsB[trainIdx].pt[1]))
                    cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
            
            cv2.imshow("Feature Matches", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result
    
    return None

# 示例用法
if __name__ == "__main__":
    # 读取两张要拼接的图像
    image1 = cv2.imread("image1.jpeg")
    image2 = cv2.imread("image2.jpeg")
    
    # 确保图像读取成功
    if image1 is None or image2 is None:
        print("无法读取图像文件")
        exit()
    
    # 调整图像大小(可选)
    image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)
    image2 = cv2.resize(image2, (0, 0), fx=0.5, fy=0.5)
    
    # 拼接图像
    stitched_image = stitch_images([image1, image2], show_matches=True)
    
    if stitched_image is not None:
        # 显示并保存结果
        cv2.imshow("Stitched Image", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("stitched_result.jpg", stitched_image)
    else:
        print("图像拼接失败，可能匹配点不足")