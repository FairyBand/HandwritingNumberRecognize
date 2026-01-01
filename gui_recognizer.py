import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
# 1. 加载训练好的模型
model = tf.keras.models.load_model('mnist_model.h5')
class DigitRecognizerGUI:
def __init__(self, root):
self.root = root
self.root.title("手写数字实时识别 演示")
# 创建画布 (280x280，是 MNIST 尺寸的 10 倍，方便书写)
self.canvas = tk.Canvas(root, width=280, height=280, bg='black', cursor="cross")
self.canvas.grid(row=0, column=0, pady=10, padx=10, columnspan=2)
# 创建 Pillow 图像对象用于后台绘制
self.image = Image.new("L", (280, 280), 0)
self.draw = ImageDraw.Draw(self.image)
# 结果显示标签
self.label_res = tk.Label(root, text="请在黑色区域写字", font=("Helvetica", 18))
self.label_res.grid(row=1, column=0, columnspan=2)
# 按钮
self.btn_clear = tk.Button(root, text="清除画布", command=self.clear_canvas)
self.btn_clear.grid(row=2, column=0, pady=10)
self.btn_predict = tk.Button(root, text="识别数字", command=self.recognize)
self.btn_predict.grid(row=2, column=1, pady=10)
# 绑定鼠标事件
self.canvas.bind("<B1-Motion>", self.paint)
def paint(self, event):
# 在画布和 Pillow 图像上同时绘制
x1, y1 = (event.x - 10), (event.y - 10)
x2, y2 = (event.x + 10), (event.y + 10)
self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
self.draw.ellipse([x1, y1, x2, y2], fill=255)
def clear_canvas(self):
self.canvas.delete("all")
self.image = Image.new("L", (280, 280), 0)
self.draw = ImageDraw.Draw(self.image)
self.label_res.config(text="请在黑色区域写字")
def recognize(self):
# 1. 将 280x280 的图像缩小为 28x28 (MNIST 标准尺寸)
img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
# 2. 转换为 numpy 数组并进行归一化
img_array = np.array(img_resized).reshape(1, 28, 28, 1).astype('float32') / 255
# 3. 模型预测
prediction = model.predict(img_array)
result = np.argmax(prediction)
confidence = np.max(prediction)
# 4. 更新界面
self.label_res.config(text=f"识别结果: {result} (置信度: {confidence:.2%})")
# 启动程序
if __name__ == "__main__":
root = tk.Tk()
app = DigitRecognizerGUI(root)
root.mainloop()
