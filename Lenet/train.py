import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from model import LeNet

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=2)
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=10000,shuffle=False,num_workers=2)
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
testdata_iter=iter(testloader)
test_images,test_labels=next(testdata_iter)
# Use below code first u need to do is change 10000 to 4 in order to show test img and label
# def imgshow(img):
#     img = img / 2 + 0.5 # unnormalize
#     npimg=img.numpy() # convert to numpy
#     plt.imshow(np.transpose(npimg,(1,2,0))) 
#     plt.show()
# # show test img an label
# print(''.join('%5s'%classes[test_labels[j]] for j in range(4)))
# imgshow(torchvision.utils.make_grid(test_images))
net=LeNet()
criterion=nn.CrossEntropyLoss()
# optimizer=torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9) # SGD optimizer result is 0.600 accuracy
optimizer=torch.optim.Adam(net.parameters(),lr=0.001) # Adam optimizer
for epoch in range(5):
    running_loss=0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels=data
        optimizer.zero_grad() # zero the parameter gradients
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step() # Does the update
        running_loss+=loss.item()
        if i%500==499:
            with torch.no_grad(): # 不要计算误差梯度
                outputs=net(test_images) # [batch_size,num_classes]
                pred_y=torch.max(outputs,1)[1] # get index of max probability
                accuracy=sum(pred_y==test_labels).item()/test_labels.size(0)
                print('[%d,%5d] loss: %.3f accuracy: %.3f' % (epoch+1,i+1,running_loss/500,accuracy))
                running_loss=0.0
print('Finished Training')
save_path='./lenet.pth'
torch.save(net.state_dict(),save_path)
print('Model saved to %s' % save_path)