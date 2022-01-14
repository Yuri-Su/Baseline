# -*- coding = utf-8 -*-
# @Time : 2021/11/24 3:23 下午
# @Author : Yuri Su
import tensorflow as tf
from tensorflow.keras import layers


class MSFF(layers.Layer):
    def __init__(self, channels=64, r=4):
        super(MSFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = tf.keras.Sequential([
            layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization()
        ])

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.global_att = tf.keras.Sequential([
            layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization()
        ])
        self.sigmoid = tf.nn.sigmoid

    def call(self, inputs, residual=None, **kwargs):
        x_fusion = inputs + residual
        xl = self.local_att(x_fusion)
        xg = self.avg_pool(x_fusion)
        xg = tf.reshape(xg, [-1, 1, 1, xg.shape[1]])
        xg = self.global_att(xg)
        weight = self.sigmoid(xl + xg)
        out = 2 * weight * inputs + 2 * residual * (1 - weight)
        return out

# if __name__ == '__main__':
#     x, residual = tf.ones([8, 32, 32, 64]), tf.ones([8, 32, 32, 64])
#     channels = x.shape[3]
#     print("channels:%d" % channels)
#     model = MSFF(channels=channels)
#     out = model(x, residual)
#     print(out.shape)
