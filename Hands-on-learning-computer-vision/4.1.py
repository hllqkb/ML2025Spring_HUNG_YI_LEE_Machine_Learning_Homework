import cv2
import numpy as np

# 读取图像和模板
img = cv2.imread('image.png')
templ = cv2.imread('template2.png')
# 检查图像是否成功加载
if img is None or templ is None:
    print("Error: Could not load one or both images")
    exit()

# 打印图像尺寸
print("Image size:", img.shape)
print("Template size:", templ.shape)
if img.shape[0] < templ.shape[0] or img.shape[1] < templ.shape[1]:
    print("Error: Template is larger than image")
    exit()
# 获取模板的高度和宽度
height, width = templ.shape[:2]

# 使用归一化平方差匹配方法
result = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)

# 设置匹配阈值
threshold = 0.7

# 获取所有大于阈值的匹配位置
locations = np.where(result >= threshold)
locations = list(zip(*locations[::-1]))  # 转换为(x,y)坐标列表

# 使用非极大值抑制(NMS)来避免重叠的矩形框
def non_max_suppression(boxes, overlap_thresh=0.5):
    if len(boxes) == 0:
        return []
    
    # 将边界框坐标转换为float类型
    boxes = np.array(boxes, dtype="float")
    
    # 初始化选择的索引列表
    pick = []
    
    # 获取边界框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 计算边界框的面积并排序
    area = (x2 - x1 ) * (y2 - y1)
    idxs = np.argsort(y2)
    
    # 循环遍历排序后的索引
    while len(idxs) > 0:
        # 获取最后一个索引并将其添加到选择列表中
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # 找到当前框与其他框的最大(x,y)坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # 计算重叠区域的宽度和高度
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1 )
        
        # 计算重叠比例
        overlap = (w * h) / area[idxs[:last]]
        
        # 删除重叠比例大于阈值的所有索引
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    # 返回选择的边界框
    return boxes[pick].astype("int")

# 创建边界框列表
rectangles = []
for loc in locations:
    rect = [loc[0], loc[1], loc[0] + width, loc[1] + height]
    rectangles.append(rect)

# 应用非极大值抑制
rectangles = non_max_suppression(rectangles)

# 绘制所有匹配的矩形框
for (x1, y1, x2, y2) in rectangles:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
while True:
    cv2.imshow('Multi-Template Matching', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
# 显示结果
cv2.destroyAllWindows()
