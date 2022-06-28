# -*- coding = utf-8 -*-
# @Time : 2021/12/9 2:50 PM
# @Author : Yuri Su

import tensorflow as tf
from tensorflow.keras import layers


class SEModule(layers.Layer):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.channels = channels
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Conv2D(filters=channels // reduction, kernel_size=1, padding='valid', activation='relu')
        self.fc2 = layers.Conv2D(filters=channels, kernel_size=1, padding='valid')
        self.sigmoid = tf.nn.sigmoid

    def call(self, inputs, *args, **kwargs):
        out = self.avg_pool(inputs)
        out = tf.reshape(out, [-1, 1, 1, self.channels])

        out = self.fc1(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        output = tf.multiply(inputs, weight)
        return output


if __name__ == '__main__':
    x = layers.Input([26, 26, 512])
    y = SEModule(512)(x)
    print(y.shape)