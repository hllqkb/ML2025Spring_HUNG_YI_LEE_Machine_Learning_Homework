import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
from model import mobilenetv3_large, mobilenetv3_small

def load_image(image_path):
    """加载并预处理图像"""
    # 数据预处理 - 与训练时的验证集预处理保持一致
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    image = data_transform(image)
    
    # 增加batch维度
    image = torch.unsqueeze(image, 0)
    
    return image

def predict_single_image(model_path, image_path, class_indices_path):
    """预测单张图像"""
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载类别索引
    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
    except FileNotFoundError:
        print(f"找不到类别索引文件: {class_indices_path}")
        return None
    
    num_classes = len(class_indices)
    
    # 创建模型
    model = mobilenetv3_large(num_classes=num_classes)  # 根据训练时使用的模型选择
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"找不到模型文件: {model_path}")
        return None
    
    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 加载和预处理图像
    try:
        image = load_image(image_path)
        image = image.to(device)
    except FileNotFoundError:
        print(f"找不到图像文件: {image_path}")
        return None
    
    # 预测
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_indices[str(predicted.item())]
        confidence_score = confidence.item()
    
    return predicted_class, confidence_score, probabilities[0]

def predict_with_top_k(model_path, image_path, class_indices_path, k=5):
    """预测并返回Top-K结果"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载类别索引
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    num_classes = len(class_indices)
    
    # 创建和加载模型
    model = mobilenetv3_large(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 预处理图像
    image = load_image(image_path).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        
        # 获取Top-K结果
        top_k_probs, top_k_indices = torch.topk(probabilities, k)
        
        results = []
        for i in range(k):
            class_name = class_indices[str(top_k_indices[0][i].item())]
            prob = top_k_probs[0][i].item()
            results.append((class_name, prob))
    
    return results

def visualize_prediction(image_path, predicted_class, confidence, save_path=None):
    """可视化预测结果"""
    # 加载原始图像
    image = Image.open(image_path)
    
    # 创建图像显示
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.6, f'预测类别: {predicted_class}', 
             fontsize=16, ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.4, f'置信度: {confidence:.4f}', 
             fontsize=14, ha='center', transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"结果已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数 - 演示预测功能"""
    # 文件路径配置
    model_path = './mobilenetv3_large.pth'
    class_indices_path = './class_indices.json'
    image_path = './test_image.jpg'  # 请替换为实际的测试图像路径
    
    print("=" * 50)
    print("MobileNet V3 图像分类预测")
    print("=" * 50)
    
    # 单张图像预测
    result = predict_single_image(model_path, image_path, class_indices_path)
    
    if result:
        predicted_class, confidence, _ = result
        print(f"\n预测结果:")
        print(f"类别: {predicted_class}")
        print(f"置信度: {confidence:.4f}")
        
        # 可视化结果
        visualize_prediction(image_path, predicted_class, confidence)
        
        # Top-5预测
        print(f"\nTop-5 预测结果:")
        top_k_results = predict_with_top_k(model_path, image_path, class_indices_path, k=5)
        for i, (class_name, prob) in enumerate(top_k_results, 1):
            print(f"{i}. {class_name}: {prob:.4f}")

if __name__ == '__main__':
    main()