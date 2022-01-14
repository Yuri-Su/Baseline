# -*- coding = utf-8 -*-
# @Time : 2021/12/9 4:25 PM
# @Author : Yuri Su
import keras
import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttentionBlock(layers.Layer):
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionBlock, self).__init__()
        self.channels = channels
        # 通道注意力
        self.share_dense1 = layers.Dense(channels // reduction, activation='relu')
        self.share_dense2 = layers.Dense(channels)
        self.maxpool = layers.GlobalMaxPool2D()
        self.avgpool = layers.GlobalAveragePooling2D()

    def call(self, inputs, *args, **kwargs):
        # 通道注意力
        out_max = self.maxpool(inputs)
        out_max = tf.reshape(out_max, [1, 1, self.channels])
        out_max = self.share_dense1(out_max)
        out_max = self.share_dense2(out_max)
        out_avg = self.avgpool(inputs)
        out_avg = tf.reshape(out_avg, [1, 1, self.channels])
        out_avg = self.share_dense1(out_avg)
        out_avg = self.share_dense2(out_avg)
        out = layers.Add()([out_avg, out_max])
        weight = tf.nn.sigmoid(out)
        out_channel = tf.multiply(inputs, weight)
        # 空间注意力
        max_pool = tf.reduce_max(out_channel, axis=-1, keepdims=True)
        avg_pool = tf.reduce_mean(out_channel, axis=-1, keepdims=True)
        out = tf.concat([max_pool, avg_pool], axis=-1)
        out = layers.Conv2D(filters=1, kernel_size=7, padding='same', kernel_initializer='he_normal')(out)
        weight_spatial = tf.nn.sigmoid(out)
        return tf.multiply(out_channel, weight_spatial)


if __name__ == '__main__':
    x = layers.Input([26, 26, 512])
    y = ChannelAttentionBlock(512)(x)
    model = tf.keras.Model(x, y)
    print(model.summary())
