import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import json
from model import AlexNet
from PIL import Image

datatransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载图像
img = Image.open('test.jpg')
plt.imshow(img)
# plt.show()
# 转换为 tensor
img = datatransform(img)
# 增加 batch 维度
img = img.unsqueeze(0)

class_indict = None
try:
    # 加载 json 文件
    with open('flower_dict.json', 'r') as f:
        class_indict = json.load(f)
    # 加载模型
    model = AlexNet(num_classes=5, init_weights=True)
    model.load_state_dict(torch.load('Alexnet_model.pth'))
    # 关闭 dropout
    model.eval()
    # 预测
    with torch.no_grad():
        output = torch.squeeze(model(img))
        probabilities = torch.nn.functional.softmax(output, dim=0)
        predict = torch.argmax(probabilities, dim=0)
        print(class_indict[str(predict.item())], probabilities[predict].item())
    plt.show()
        
except Exception as e:
    print(e)
