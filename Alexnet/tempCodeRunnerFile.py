trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=4)
# testset=datasets.ImageFolder(root='./flower_data/val',transform=datatransform['val'])
# testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=4)

# model=AlexNet(num_classes=102,init_weights=True)
# model.to(device)
# criterion=nn.CrossEntropyLoss()
# optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

# for epoch in range(20):
#     running_loss=0.0
#     for i,data in enumerate(trainloader,0):
#         inputs,labels=data
#         inputs,labels=inputs.to(device),labels.to(device)
#         optimizer.zero_grad()
#         outputs=model(inputs)
#         loss=criterion(outputs,labels)
#         loss.backward()
#         optimizer.step()
#         running_loss+=loss.item()
#         if i%2000==1999:
#             print('[%d, %5d] loss: %.3f' %(epoch+1,i+1,running_loss/2000))
#             running_loss=0.0

# print('Finished Training')

# correct=0
# total=0
# with torch.no_grad():
#     for data in testloader:
#         images,labels=data
#         images,labels=images.to(device),labels.to(device)
#         outputs=model(images)
#         _,predicted=torch.max(outputs.data,1)
#         total+=labels.size(0)
#         correct+=((predicted==labels).sum().item())

# print('Accuracy of the network on the test images: %d %%' % (100*correct/total))