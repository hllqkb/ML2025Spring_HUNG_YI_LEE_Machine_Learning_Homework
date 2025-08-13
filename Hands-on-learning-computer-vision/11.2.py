import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import VOCSegmentation
from torchvision import transforms as T
import torch.nn as nn

# -------------------
# 配置
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 21
DATA_DIR = "/home/hllqk/projects/dive-into-deep-learning/Hands-on-learning-computer-vision/"

VOC_COLORMAP = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
    (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128)
]
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
# 数据集类
# -------------------
class VOCSegDataset(VOCSegmentation):
    def __init__(self, root, year, image_set, transforms_img=None, transforms_mask=None):
        super().__init__(root=root, year=year, image_set=image_set, download=False)
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
val_img_tf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_mask_tf = T.Compose([
    T.Resize((256, 256), interpolation=Image.NEAREST),
    T.PILToTensor()
])

val_set = VOCSegDataset(DATA_DIR, year="2012", image_set="val",
                        transforms_img=val_img_tf, transforms_mask=val_mask_tf)

# -------------------
# 加载模型
# -------------------
model = FCN8s(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# -------------------
# 彩色映射函数
# -------------------
def decode_segmap(mask):
    r = np.zeros_like(mask, dtype=np.uint8)
    g = np.zeros_like(mask, dtype=np.uint8)
    b = np.zeros_like(mask, dtype=np.uint8)
    for l in range(NUM_CLASSES):
        idx = mask == l
        r[idx], g[idx], b[idx] = VOC_COLORMAP[l]
    return np.stack([r, g, b], axis=2)

# -------------------
# 预测函数
# -------------------
def predict(img_tensor):
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # batch=1
    with torch.no_grad():
        out = model(img_tensor)
        pred = out.argmax(1).squeeze(0).cpu().numpy()
    return decode_segmap(pred)

# -------------------
# 可视化预测
# -------------------
num_image = 10
_, figs = plt.subplots(num_image, 3, figsize=(12, 22))

for i in range(num_image):
    img, mask = val_set[i]
    mask = mask.squeeze(0).numpy()
    pred_color = predict(img)
    mask_color = decode_segmap(mask)

    img_pil = T.ToPILImage()(img)
    
    figs[i, 0].imshow(img_pil)
    figs[i, 0].axis('off')
    figs[i, 1].imshow(mask_color)
    figs[i, 1].axis('off')
    figs[i, 2].imshow(pred_color)
    figs[i, 2].axis('off')

# 添加标题
figs[num_image-1, 0].set_title("Image", y=-0.2)
figs[num_image-1, 1].set_title("Label", y=-0.2)
figs[num_image-1, 2].set_title("FCN8s", y=-0.2)

plt.savefig("predict_8s.png")
plt.show()
