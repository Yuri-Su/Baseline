import numpy as np
import matplotlib.pyplot as plt
import tensorflow.python.keras as keras
from tensorflow.python.keras import layers
from tensorflow.keras.datasets import mnist

# print(tf.__version__) -> 2.1.0

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(np.max(x_train)) -> 255
# print(x_train.shape) -> 60000,28,28

# 归一化
x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5,batch_size=128)
res = model.evaluate(x_test, y_test)  # 返回损失和精度
print(model.metrics_names)
print(res)

# 随机选取图片测试
img_random = x_test[np.random.randint(0, len(x_test))]
plt.imshow(img_random)
plt.show()

# 模型预测
img_random = (np.expand_dims(img_random, 0))
prob = model.predict(img_random)
print(np.argmax(prob))
