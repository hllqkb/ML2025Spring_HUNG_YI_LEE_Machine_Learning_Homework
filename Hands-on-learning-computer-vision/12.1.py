import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt

# 加载预训练的 Faster R-CNN 模型（基于 MS COCO 数据集）
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 21  # 20 个类别 + 背景
in_features = model.roi_heads.box_predictor.cls_score.in_features
# 替换模型的预测头以匹配自定义类别数
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 将模型移动到 GPU 或 CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 定义目标检测函数
def obj_detect(img_path):
    # 读取图像并转换为 RGB 格式
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # 归一化图像像素值到 [0, 1] 范围
    img /= 255.0
    # 将图像转换为 PyTorch 张量（HWC 转 CHW 格式）
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 形状: [1, 3, H, W]
    # 将图像张量移动到指定设备（GPU 或 CPU）
    img = img.to(device)
    # 设置模型为评估模式
    model.eval()
    # 设置置信度阈值，过滤低置信度的预测
    threshold = 0.2
    # 执行预测（禁用梯度计算以节省内存）
    with torch.no_grad():
        pred = model(img)
    # 提取预测结果
    boxes = pred[0]['boxes'].cpu().numpy()  # 边界框坐标
    scores = pred[0]['scores'].cpu().numpy()  # 置信度分数
    labels = pred[0]['labels'].cpu().numpy()  # 类别标签
    # 过滤低置信度的预测
    keep = scores > threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    # 返回检测结果
    return boxes, scores, labels

# 定义 ImageNet 验证集的路径
pred_path = "ImageNet/imagenet/val/"
if os.path.exists(pred_path) == False:
    print("ImageNet 验证集路径不存在！")
    exit()
pred_files = [os.path.join(pred_path, f) for f in os.listdir(pred_path)]

# 定义类别名称字典
classes = {
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
    6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
    11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike',
    15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa',
    19: 'train', 20: 'tvmonitor'
}

# 设置绘图画布
plt.figure(figsize=(20, 60))
image_list = [0, 11, 17, 28]  # 指定要显示的图像索引

# 遍历图像文件
for i, image_path in enumerate(pred_files):
    if i > 30:  # 限制处理的图像数量
        break
    if i not in image_list:  # 跳过不需要显示的图像
        continue
    # 执行目标检测
    boxes, scores, labels = obj_detect(image_path)
    # 读取图像用于可视化
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 绘制边界框和类别标签
    for j, box in enumerate(boxes):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 220, 0), 2)
        if labels[j] in classes:
            cv2.putText(img, classes[labels[j]], (int(box[0]), int(box[1]) - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (220, 0, 0), 1, cv2.LINE_AA)
    # 显示图像
    plt.subplot(10, 2, image_list.index(i) + 1)
    plt.axis('off')
    plt.imshow(img)

plt.show()
