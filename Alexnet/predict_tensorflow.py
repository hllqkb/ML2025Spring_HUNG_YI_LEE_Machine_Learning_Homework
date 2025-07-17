
import matplotlib.pyplot as plt
import numpy as np
import json
from model_tensorflow import AlexNet_v1
from PIL import Image
import os
# 禁用 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 加载图像
img = Image.open('test.jpg')
plt.imshow(img)
# plt.show()
img=img.resize((224,224))
# scaling pixel value to (0-1)
img = np.array(img) / 255.

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
class_indict = None
try:
    # 加载 json 文件
    with open('flower_dict.json', 'r') as f:
        class_indict = json.load(f)
    # 加载模型
    model = AlexNet_v1(num_classes=5)
    model.load_weights('./save_weights/myAlex100%.weights.h5')
    # 预测
    result= np.squeeze(model.predict(img))
    pred_class = np.argmax(result)
    print(class_indict[str(pred_class)], result[pred_class])
        
except Exception as e:
    print(e)
