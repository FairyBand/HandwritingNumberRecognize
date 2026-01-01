import tensorflow as tf
from tensorflow.keras import layers, models
# 1. 加载数据集 (MNIST)
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 2. 数据预处理：归一化并调整维度
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
# 3. 构建 CNN 模型
model = models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Flatten(),
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax') # 10 个数字分类
])
# 4. 编译和训练
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
print("开始训练...")
model.fit(train_images, train_labels, epochs=5, batch_size=64)
# 5. 保存模型
model.save('mnist_model.h5')
print("模型已保存为 mnist_model.h5")
