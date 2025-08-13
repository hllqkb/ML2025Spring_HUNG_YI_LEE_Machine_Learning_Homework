import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os

# -------------------
# 1. 配置
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 21  # VOC2012: 20类 + 背景
BATCH_SIZE = 4 # 批大小
EPOCHS = 20  # 训练轮数
LR = 1e-4  # 学习率
# 应该是 VOCdevkit 的父目录，注意！即不包含 VOCdevkit 的目录
DATA_DIR = "/home/hllqk/projects/dive-into-deep-learning/Hands-on-learning-computer-vision/"

# VOC 官方调色板（21类）
VOC_COLORMAP = [
    (0, 0, 0),        # 背景
    (128, 0, 0),      # aeroplane
    (0, 128, 0),      # bicycle
    (128, 128, 0),    # bird
    (0, 0, 128),      # boat
    (128, 0, 128),    # bottle
    (0, 128, 128),    # bus
    (128, 128, 128),  # car
    (64, 0, 0),       # cat
    (192, 0, 0),      # chair
    (64, 128, 0),     # cow
    (192, 128, 0),    # diningtable
    (64, 0, 128),     # dog
    (192, 0, 128),    # horse
    (64, 128, 128),   # motorbike
    (192, 128, 128),  # person
    (0, 64, 0),       # potted plant
    (128, 64, 0),     # sheep
    (0, 192, 0),      # sofa
    (128, 192, 0),    # train
    (0, 64, 128)      # tv/monitor
]

# -------------------
# 2. 数据集
# -------------------
class VOCSegDataset(VOCSegmentation):
    def __init__(self, root, year, image_set, transforms_img=None, transforms_mask=None):
        super().__init__(root=root, year=year, image_set=image_set, download=False) # 允许下载VOC2012数据集
        self.transforms_img = transforms_img
        self.transforms_mask = transforms_mask

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.transforms_img:
            img = self.transforms_img(img)
        if self.transforms_mask:
            target = self.transforms_mask(target)
        return img, target

# 图像变换
train_img_tf = T.Compose([
    T.Resize((256, 256)),          # 调整图像大小
    T.ToTensor(),                  # 转为张量并归一化到 [0, 1]
    T.Normalize(                   # 标准化（基于ImageNet统计量）
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
train_mask_tf = T.Compose([
    T.Resize((256, 256), interpolation=Image.NEAREST),  # 最近邻插值（避免引入无效类别）
    T.PILToTensor()                                     # 转为张量（保持整数标签）
])

train_set = VOCSegDataset(DATA_DIR, year="2012", image_set="train",
                          transforms_img=train_img_tf,
                          transforms_mask=train_mask_tf)
val_set = VOCSegDataset(DATA_DIR, year="2012", image_set="val",
                        transforms_img=train_img_tf,
                        transforms_mask=train_mask_tf)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# -------------------
# 3. FCN 网络 (基于 VGG16)
# -------------------
class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        
        self.stage1 = nn.Sequential(*features[:17])  # 到 pool3
        self.stage2 = nn.Sequential(*features[17:24]) # 到 pool4
        self.stage3 = nn.Sequential(*features[24:])   # 到 pool5
        
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.score_final = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1, bias=False)
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4, bias=False)

    def forward(self, x):
        pool3 = self.stage1(x)
        pool4 = self.stage2(pool3)
        pool5 = self.stage3(pool4)
        
        score_pool3 = self.score_pool3(pool3)
        score_pool4 = self.score_pool4(pool4)
        score_final = self.score_final(pool5)
        
        upscore2 = self.upsample_2x(score_final)  # x2
        fuse_pool4 = upscore2 + score_pool4
        upscore_pool4 = self.upsample_2x(fuse_pool4)  # x2
        fuse_pool3 = upscore_pool4 + score_pool3
        out = self.upsample_8x(fuse_pool3)  # x8
        
        return out

# -------------------
# 4. 训练
# -------------------
model = FCN8s(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=255)  # VOC 标签中 255 表示忽略
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")  # 记录最佳模型

# 将预测结果转换为 VOC 彩色图
def decode_segmap(mask):
    r = np.zeros_like(mask, dtype=np.uint8)
    g = np.zeros_like(mask, dtype=np.uint8)
    b = np.zeros_like(mask, dtype=np.uint8)
    for l in range(NUM_CLASSES):
        idx = mask == l
        r[idx], g[idx], b[idx] = VOC_COLORMAP[l]
    return np.stack([r, g, b], axis=2)

def train():
    global best_val_loss
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.squeeze(1).long().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        val_loss = validate(epoch)

        print(f"Epoch [{epoch+1}/{EPOCHS}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ 保存最佳模型 (Val Loss: {best_val_loss:.4f})")

def validate(epoch):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(val_loader):
            imgs = imgs.to(DEVICE)
            masks = masks.squeeze(1).long().to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_val_loss += loss.item()

            if i == 0:  # 保存第一个batch的预测图
                preds = outputs.argmax(1).cpu().numpy()[0]
                color_pred = decode_segmap(preds)
                Image.fromarray(color_pred).save(f"pred_epoch{epoch}.png")
    return total_val_loss / len(val_loader)

if __name__ == "__main__":
    train()
