import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import numpy as np

# 加载模型和类名
from model import LeNet

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = LeNet()
net.load_state_dict(torch.load('./lenet.pth'))
net.eval()  # 设置模型为评估模式

# 创建Streamlit界面
st.title("LeNet 图像分类器")
st.write("请上传一张图片，模型将预测其类别。")

# 上传图片
uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 打开并显示上传的图片
    image = Image.open(uploaded_file)
    st.image(image, caption='上传的图片', use_column_width=True)

    # 对图片进行预处理
    image = transform(image)
    image = torch.unsqueeze(image, 0)  # 添加批次维度 [B, C, H, W]

    # 进行预测
    with torch.no_grad():
        output = net(image)
        pred = torch.max(output, 1)[1].data.numpy()
        pred_probabilities = torch.nn.functional.softmax(output, dim=1).data.numpy()[0]

    # 输出预测的类别
    predicted_class = classes[int(pred[0])]
    st.write(f"预测的类别: {predicted_class}")

    # 输出所有类别的概率
    st.write("每个类别的概率:")
    for class_name, prob in zip(classes, pred_probabilities):
        st.write(f"{class_name}: {prob * 100:.2f}%")

    # 输出概率最高的前N个类别（例如前3个）
    top_n = 3
    top_indices = torch.argsort(output, dim=1, descending=True)[0][:top_n]
    top_classes = [classes[int(idx)] for idx in top_indices]
    top_probabilities = [pred_probabilities[int(idx)] for idx in top_indices]

    st.write(f"\n前 {top_n} 个预测类别及其概率:")
    for class_name, prob in zip(top_classes, top_probabilities):
        st.write(f"{class_name}: {prob * 100:.2f}%")

    # 绘制概率条形图
    st.bar_chart(pred_probabilities)
