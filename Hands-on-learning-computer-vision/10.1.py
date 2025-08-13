# 图像分类
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from imutils import paths
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
# 进行训练集数据增强
train_transformer = transforms.Compose([
    transforms.ToPILImage(),                  # 将numpy数组转换为PIL格式
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 新增颜色增强
    # transforms.RandomHorizontalFlip(p=0.5),          # 随机水平翻转
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # 随机缩放并裁剪到 224x224
    transforms.ToTensor(),                     # 转换为Tensor
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),  # 归一化
])
# 验证集通常不包含数据增强操作（如随机裁剪、翻转等）但是要保持相同的图像大小(224x224)和归一化参数
val_transformer = transforms.Compose([
    transforms.ToPILImage(),                  # 将numpy数组转换为PIL格式
    transforms.Resize((224, 224)),             # 调整图像大小（与训练集相同）
    transforms.ToTensor(),                    # 转换为Tensor
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),  # 归一化
    ])
# 图像数据
data=[]
# 标签数据
labels=[]
# 储存标签的临时变量
labels_tep=[]
# 获取caltech101文件夹下的全部图片的列表
image_paths=list(paths.list_images('./caltech-101/'))
# ./caltech-101/101_ObjectCategories/elephant/image_0001.jpg
# ./caltech-101/101_ObjectCategories/elephant/image_0057.jpg
step=0
for image_path in image_paths:
    # 获取倒数第二个'/'前的文本也就是label elephant
    label=image_path.split(os.path.sep)[-2]
    # 读取
    image=cv2.imread(image_path)
    # 将图像从BGR转向RGB OpenCV默认使用BGR格式读取图像，而大多数深度学习框架(如TensorFlow/PyTorch)使用RGB格式
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # 统一输入图像尺寸 使用INTER_AREA插值方法，适合缩小图像时保持清晰度
    image=cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA)
    data.append(image)
    labels_tep.append(label)

# 构建标签映射
unique_labels = sorted(set(labels_tep))
name2label = {label: idx for idx, label in enumerate(unique_labels)}
# print(name2label)
# 生成数字标签
labels = [name2label[label] for label in labels_tep]
# print(labels)
# 转成numpy数组
data=np.array(data)
labels=np.array(labels)
# 将数据划分成训练集和验证集和测试集
# X_train和y_train：60%的训练数据 其他数据用于验证和测试40%
X_train,X,y_train,Y=train_test_split(data,labels,test_size=0.4,random_state=42)
# X_val和y_val：20%的验证数据 另外20%用于测试
X_val,X_test,y_val,y_test=train_test_split(X,Y,test_size=0.5,random_state=42)
# (5486, 224, 224, 3) (1829, 224, 224, 3) (1829, 224, 224, 3)
print(X_train.shape,X_val.shape,X_test.shape)
# 把caltech101数据集封装成Dataset类
class Caltech101Dataset(Dataset):
    def __init__(self, image,label=None,transform=None):
        self.image = image
        self.label = label
        self.transform = transform
    def __len__(self):
        return len(self.image)
    def __getitem__(self, index):
        img = self.image[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.label is not None:
            label = self.label[index]
            return img,label
        else:
            return img

# 生成不同类用于训练和验证和测试
train_dataset =Caltech101Dataset(X_train,y_train,transform=train_transformer)
val_dataset =Caltech101Dataset(X_val,y_val,transform=val_transformer)
test_dataset =Caltech101Dataset(X_test,y_test,transform=val_transformer)

# # 冻结参数(可选)，冻结后训练速度更快，但是准确率会下降
# # for param in model.parameters():
# #     param.requires_grad = False
# 加载ResNet34模型
model = torchvision.models.resnet34(pretrained='imagenet')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 把模型转移到GPU上
model.to(device)
# 定义训练函数
def fit(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_top1_error = 0
    
    for i, data in enumerate(dataloader):
        x, y = data[0].to(device), data[1].to(device)
        # 梯度清零
        optimizer.zero_grad()
        outputs = model(x)
        # 计算损失
        loss = criterion(outputs, y)
        
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_top1_error += torch.sum(preds != y).item()
        
        loss.backward()
        optimizer.step()
    
    loss = running_loss / len(dataloader.dataset)
    top1_error = running_top1_error / len(dataloader.dataset)
    print(f"Train Loss: {loss:.4f}, Train Top-1 Error: {top1_error:.4f}")
    return loss, top1_error

# 定义验证函数
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_top1_error = 0
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x, y = data[0].to(device), data[1].to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_top1_error += torch.sum(preds != y).item()
    
    loss = running_loss / len(dataloader.dataset)
    top1_error = running_top1_error / len(dataloader.dataset)
    print(f'Val Loss: {loss:.4f}, Val Top-1 Error: {top1_error:.4f}')
    return loss, top1_error

# 定义测试函数
def test(model, dataloader):
    top1_error = 0
    total = 0
    
    with torch.no_grad():
        for data in dataloader:
            x, y = data[0].to(device), data[1].to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            top1_error += torch.sum(predicted != y).item()
    
    return top1_error, total

# 主训练流程
def main():
    # 每次传入批次大小，不建议太大
    BATCH_SIZE = 64
    # 训练轮数
    epochs = 25
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 模型和优化器初始化
    model = torchvision.models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(unique_labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    

    train_loss, train_top1_error = [], []
    val_loss, val_top1_error = [], []
    
    print(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)} examples...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1} of {epochs}")
        
        train_epoch_loss, train_epoch_top1_error = fit(model, trainloader, optimizer, criterion)
        val_epoch_loss, val_epoch_top1_error = validate(model, valloader, criterion)
        
        train_loss.append(train_epoch_loss)
        train_top1_error.append(train_epoch_top1_error)
        val_loss.append(val_epoch_loss)
        val_top1_error.append(val_epoch_top1_error)
        
        torch.save(model.state_dict(), "model.pth")
    
    # 绘制曲线
    plt.figure(figsize=(10, 7))
    plt.plot(train_top1_error, label='train top-1 error')
    plt.plot(val_top1_error, label='validation top-1 error')
    plt.xlabel('Epoch')
    plt.ylabel('top-1 Error')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    top1_error, total = test(model, testloader)
    print(f'Top-1 Error of the network on test images: {100 * top1_error / total:.3f}%')

if __name__ == "__main__":
    main()