from torchvision import datasets, transforms, utils
import torch
import torch.nn as nn
import torch.optim as optim
import os, json, time
import numpy as np
import torchvision
from model import AlexNet
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 进行数据增强
datatransform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪到224*224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转为tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  # 先将图片resize到256*256
        transforms.CenterCrop(224),  # 然后中心裁剪到224*224
        transforms.ToTensor(),  # 转为tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])
}

trainset = datasets.ImageFolder(root='./flower_data/train', transform=datatransform['train'])
print(len(trainset))
flower_list = trainset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open('flower_dict.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = datasets.ImageFolder(root='./flower_data/val', transform=datatransform['val'])
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=4)

# testdata_iter = iter(testloader)
# images, labels = next(testdata_iter)

# def showimg(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# print(' '.join('%5s' % cla_dict[labels[j].item()] for j in range(4)))
# showimg(utils.make_grid(images))

# 定义AlexNet模型
model = AlexNet(num_classes=5, init_weights=True)
model.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0002)
save_path='./Alexnet_model.pth'
best_acc = 0.0
for epoch in range(40):
    model.train()
    t1=time.perf_counter()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print train process
        rate=(i+1)/len(trainloader)
        a="*" * int(rate*50)
        b="." * (50-int(rate*50))
        print("\rtrain epoch[{}/{}] loss:{:.3f} [{}{}]".format(epoch+1,20,running_loss/(i+1),a,b),end="")
    print()
    print("train epoch[{}/{}] loss:{:.3f} time:{:.2f}s".format(epoch+1,20,running_loss/(i+1),time.perf_counter()-t1))
    model.eval()
    acc=0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            acc += (predicted == labels).sum().item()
    acc /= len(testset)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), save_path)
    print('test accuracy: {:.2f} %'.format(acc*100))

print('Finished Training')
# the best test accuracy is about 80 %