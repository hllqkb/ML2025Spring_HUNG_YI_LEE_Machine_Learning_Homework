from PIL import Image
import numpy as np
import torch
import json
from matplotlib import pyplot as plt
from torchvision import transforms
from model import GoogLeNet
data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img=Image.open('test.jpg')
plt.imshow(img)
img_tensor = data_transform(img)
class_indict=None
img_tensor = img_tensor.unsqueeze(0)
try:
    with open('class_indices.json', 'r') as f:
        class_indict = json.load(f)
except FileNotFoundError:
    print('class_indices.json not found')
model = GoogLeNet(aux_logits=False, num_classes=5)
missing_keys, unexpected_keys = model.load_state_dict(torch.load('./googleNet.pth'), strict=False)
model.eval()
with torch.no_grad():
    output=torch.squeeze(model(img_tensor)) # 移除张量中所有长度为1的维度
    predict=torch.softmax(output,dim=0) # 计算softmax值
    predict_cla=torch.argmax(predict).numpy() # 取最大值对应的索引
    print( {'class':class_indict[str(predict_cla)], 'prob':str(predict.max().numpy())} ) # 返回类别和预测概率
    # plt.show()