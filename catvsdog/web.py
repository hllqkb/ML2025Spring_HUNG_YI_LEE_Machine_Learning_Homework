import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# 完全禁用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 确保TensorFlow使用CPU
tf.config.set_visible_devices([], 'GPU')

# 页面配置
st.set_page_config(
    page_title="🐱🐶 猫狗识别器",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    .cat-result {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    }
    .dog-result {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    .confidence-bar {
        background-color: rgba(255,255,255,0.3);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """加载模型（缓存）"""
    try:
        model = tf.saved_model.load("./saved_model/saved_model_dir")
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

def preprocess_image(image):
    """预处理图片 - 修正为与训练时一致"""
    # 调整大小到224x224
    image = image.resize((224, 224))
    # 转换为numpy数组
    image_array = np.array(image)
    # 如果是灰度图，转换为RGB
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    # 如果有alpha通道，移除
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    
    # 重要：不要归一化！ImageDataGenerator默认不归一化
    # 保持像素值在[0,255]范围，与训练时一致
    image_array = image_array.astype(np.float32)
    # 添加batch维度
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(model, image_array):
    """预测图片 - 修正预测逻辑"""
    try:
        infer = model.signatures["serving_default"]
        
        with tf.device('/CPU:0'):
            input_tensor = tf.constant(image_array)
            predictions = infer(input_tensor)
        
        output_key = list(predictions.keys())[0]
        result = predictions[output_key].numpy()
        
        return result
    except Exception as e:
        st.error(f"预测失败: {e}")
        return None

def main():
    # 主标题
    st.markdown('<h1 class="main-header">🐱🐶 AI猫狗识别器</h1>', unsafe_allow_html=True)
    
    # 副标题
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 1.2rem;'>上传一张图片，让AI告诉你这是猫咪还是狗狗！</p>", 
        unsafe_allow_html=True
    )
    
    # 加载模型
    model = load_model()
    if model is None:
        st.stop()
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "选择图片文件",
        type=['png', 'jpg', 'jpeg'],
        help="支持PNG、JPG、JPEG格式"
    )
    
    if uploaded_file is not None:
        # 创建两列布局
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 显示原图
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片",  use_container_width=True)
        
        with col2:
            # 预测按钮
            if st.button("🔍 开始识别", type="primary", use_container_width=True):
                with st.spinner("AI正在分析中..."):
                    # 预处理图片
                    processed_image = preprocess_image(image)
                    
                    # 进行预测
                    result = predict_image(model, processed_image)
                    
                    if result is not None:
                        # 修正：二分类sigmoid输出解析
                        confidence = float(result[0][0])  # sigmoid输出值
                        
                        # 根据训练时的类别映射确定预测结果
                        # 通常：0=cats, 1=dogs（需要确认你的数据集类别顺序）
                        if confidence > 0.5:
                            prediction = "狗狗"
                            confidence_pct = confidence * 100
                            gradient_class = "dog-result"
                            emoji = "🐶"
                        else:
                            prediction = "猫咪"
                            confidence_pct = (1 - confidence) * 100
                            gradient_class = "cat-result"
                            emoji = "🐱"
                        
                        # 显示结果
                        st.markdown(f"""
                        <div class="prediction-box {gradient_class}">
                            <h2>{emoji} 这是一只 {prediction}！</h2>
                            <div class="confidence-bar">
                                <p style="margin: 0; font-size: 1.1rem;">
                                    置信度: {confidence_pct:.1f}%
                                </p>
                                <p style="margin: 0; font-size: 0.9rem;">
                                    原始输出: {confidence:.4f}
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 置信度进度条
                        st.progress(confidence_pct / 100)
    
    # 底部信息
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888; font-size: 0.9rem;'>🤖 基于深度学习的图像识别技术</p>", 
        unsafe_allow_html=True
    )
    
    # 侧边栏信息
    with st.sidebar:
        st.markdown("### 📋 使用说明")
        st.markdown("""
        1. 点击"Browse files"上传图片
        2. 支持PNG、JPG、JPEG格式
        3. 点击"开始识别"按钮
        4. 查看AI识别结果
        """)
        
        st.markdown("### 🔧 技术栈")
        st.markdown("""
        - **深度学习**: TensorFlow
        - **模型**: EfficientNet
        - **前端**: Streamlit
        - **图像处理**: PIL
        """)
        
        st.markdown("### 📊 模型信息")
        st.markdown("""
        - **准确率**: >90%
        - **输入尺寸**: 224×224
        - **类别**: 猫、狗
        """)

if __name__ == "__main__":
    main()
