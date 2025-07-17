import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

class DrawNumber(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # 初始化变量
        self.last_x = None
        self.last_y = None
        self.brush_size = 16
        
        # 窗口配置
        self.title("MNIST Digit Recognizer")
        self.geometry("400x500")
        
        # 创建画布
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack(pady=20)
        
        # 初始化绘图图像
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_drawing)
        
        # 创建按钮
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=10)
        
        self.recognize_btn = tk.Button(
            self.button_frame, 
            text="Recognize", 
            command=self.recognize
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=10)
        
        self.clear_btn = tk.Button(
            self.button_frame, 
            text="Clear", 
            command=self.clear_canvas
        )
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # 结果显示
        self.result_label = tk.Label(self, text="Draw a digit (0-9)", font=('Arial', 14))
        self.result_label.pack(pady=20)

    def paint(self, event):
        """绘制连续线条"""
        x, y = event.x, event.y
        
        if self.last_x is not None and self.last_y is not None:
            # 在画布上画线
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=self.brush_size,
                fill='black',
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND
            )
            
            # 在PIL图像上画线
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill='black',
                width=self.brush_size
            )
        
        # 在当前位置画圆点
        self.canvas.create_oval(
            x-self.brush_size/2, y-self.brush_size/2,
            x+self.brush_size/2, y+self.brush_size/2,
            fill='black', outline='black'
        )
        
        # 更新最后位置
        self.last_x = x
        self.last_y = y

    def reset_drawing(self, event):
        """重置绘图跟踪"""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill='white')
        self.result_label.config(text="Draw a digit (0-9)")
    
    def recognize(self):
        try:
            # 反色处理（黑底白字 → 白底黑字）
            img_inverted = Image.eval(self.image, lambda x: 255 - x)
            img_resized = img_inverted.resize((28, 28))
            
            img_array = np.array(img_resized).reshape(1, 28, 28, 1)
            img_array = img_array / 255.0
            
            prediction = model.predict(img_array)
            predicted_number = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            self.result_label.config(
                text=f"Prediction: {predicted_number} (Confidence: {confidence:.1f}%)",
                foreground="#0066cc"
            )
        except Exception as e:
            self.result_label.config(
                text=f"Error: {str(e)}",
                foreground="#cc0000"
            )

# 加载模型
model = tf.keras.models.load_model('best_model_simple.keras')

# 运行应用
app = DrawNumber()
app.mainloop()