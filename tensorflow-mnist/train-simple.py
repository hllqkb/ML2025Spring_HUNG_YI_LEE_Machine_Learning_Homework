import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用 GPU，强制使用 CPU
# 加载 MNIST 数据集并预处理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化
x_train = x_train[..., tf.newaxis]  # 增加通道维度 (28, 28, 1)
x_test = x_test[..., tf.newaxis]

# 使用 Keras Sequential 模型（更简单）
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),  # 展平输入
    layers.Dense(128, activation='relu'),     # 全连接层
    layers.Dense(10, activation='softmax')    # 输出层（10个类别）
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 定义回调（自动保存最佳模型）
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model_simple.keras',  # 保存路径
        save_best_only=True,        # 只保存最优模型
        monitor='val_accuracy'      # 根据验证集准确率优化
    )
]

# 训练模型
model.fit(
    x_train, y_train,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"测试集准确率: {test_acc:.4f}")