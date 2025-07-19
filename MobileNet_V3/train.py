import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
import os
from model import mobilenetv3_large, mobilenetv3_small

def main():
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪到224x224
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 数据集路径 - 请根据实际情况修改
    data_root = "./data"  # 数据集根目录
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"),
                                       transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"),
                                     transform=data_transform["val"])

    # 获取类别数量
    num_classes = len(train_dataset.classes)
    print(f"类别数量: {num_classes}")
    
    # 保存类别索引
    class_indices = dict((val, key) for key, val in train_dataset.class_to_idx.items())
    json_str = json.dumps(class_indices, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 创建模型
    model = mobilenetv3_large(num_classes=num_classes)  # 或使用 mobilenetv3_small
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练参数
    epochs = 30
    best_acc = 0.0
    save_path = './mobilenetv3_large.pth'

    # 训练循环
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 50)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_num = 0

        start_time = time.time()
        
        for step, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            train_num += inputs.size(0)

            # 打印训练进度
            if step % 50 == 0:
                print(f'Step [{step}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {torch.sum(preds == labels.data).item()/inputs.size(0):.4f}')

        # 计算训练集准确率
        train_loss = running_loss / len(train_loader)
        train_acc = running_corrects.double() / train_num

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item()
                val_corrects += torch.sum(preds == labels.data)
                val_num += inputs.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_corrects.double() / val_num

        # 更新学习率
        scheduler.step()

        # 打印epoch结果
        epoch_time = time.time() - start_time
        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')
        print(f'训练时间: {epoch_time:.2f}s')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'保存最佳模型，验证准确率: {best_acc:.4f}')

    print(f'\n训练完成！最佳验证准确率: {best_acc:.4f}')

if __name__ == '__main__':
    main()