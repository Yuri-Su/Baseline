# -*- coding = utf-8 -*-
# @Time : 2022/1/4 10:09 AM
# @Author : Yuri Su
import tensorflow as tf

from networks.backbone.swin_transformer.swin_t import SwinTransformer

num_classes = 1000
swin_t = SwinTransformer('swin_tiny_224', num_classes=num_classes, include_top=False, pretrained=False)

x = tf.random.uniform(shape=[8, 224, 224, 3], minval=0, maxval=1)

model = tf.keras.Sequential([
    swin_t,
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
y = model.predict(x)
print("class is ", tf.argmax(y[0]))
