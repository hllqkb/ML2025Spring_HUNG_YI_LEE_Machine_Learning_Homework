import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from model import LeNet
transform = transforms.Compose(
    [transforms.Resize((32,32)),
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
net = LeNet()
net.load_state_dict(torch.load('./lenet.pth'))
# im=Image.open('./test1.png')
# im=Image.open('./test.jpg')
im=Image.open('./test3.jpg')
im=transform(im) # [C.H.W]
im=torch.unsqueeze(im,0) # [B.C.H.W]
# 进行预测
with torch.no_grad():
    output = net(im)
    pred = torch.max(output, 1)[1].data.numpy()
    pred_probabilities = torch.nn.functional.softmax(output, dim=1).data.numpy()[0]

    # 输出预测的类别
    predicted_class = classes[int(pred[0])]
    print(f"Predicted Class: {predicted_class}")

    # 输出所有类别的概率
    print("Probabilities for each class:")
    for class_name, prob in zip(classes, pred_probabilities):
        print(f"{class_name}: {prob * 100:.2f}%")

    # 输出概率最高的前N个类别（例如前3个）
    top_n = 3
    top_indices = torch.argsort(output, dim=1, descending=True)[0][:top_n]
    top_classes = [classes[int(idx)] for idx in top_indices]
    top_probabilities = [pred_probabilities[int(idx)] for idx in top_indices]

    print(f"\nTop {top_n} predicted classes and their probabilities:")
    for class_name, prob in zip(top_classes, top_probabilities):
        print(f"{class_name}: {prob * 100:.2f}%")