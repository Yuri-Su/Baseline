# -*- coding = utf-8 -*-
# @Time : 2022/1/5 3:24 PM
# @Author : Yuri Su

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from networks.backbone.SEModule.SE_Module import SEModule

cfgs = {
    'rexnet1_0x': {'width_mult': 1.0, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_0x_224-ab7b9733.pth'},
    'rexnet1_3x': {'width_mult': 1.3, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_3x_224-95479104.pth'},
    'rexnet1_5x': {'width_mult': 1.5, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_5x_224-c42a16ac.pth'},
    'rexnet2_0x': {'width_mult': 2.0, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet2_0x_224-c8802402.pth'},
    'rexnet2_2x': {'width_mult': 2.2, 'depth_mult': 1.0,
                   'url': None},
}


class SiLU(layers.Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return inputs * tf.nn.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class ReLU6(layers.Layer):
    def __init__(self):
        super(ReLU6, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return tf.nn.relu6(inputs)


class ReXBlock(layers.Layer):
    def __init__(self, in_channels, channels, t, stride, use_se=True, se_ratio=12, act_layer=None, drop_layer=None):
        super(ReXBlock, self).__init__()
        if act_layer is None:
            act_layer = ReLU6()
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.channels = channels

        _layer = []
        if t != 1:
            dw_channels = in_channels * t
            _layer.extend([
                layers.Conv2D(filters=dw_channels, kernel_size=1, strides=1),
                SiLU(),
                layers.BatchNormalization(),
                layers.Dropout(drop_layer)])
        else:
            dw_channels = in_channels

        _layer.extend([
            layers.Conv2D(filters=dw_channels, kernel_size=3, padding='same', strides=stride, groups=dw_channels),
            layers.BatchNormalization(),
            layers.Dropout(drop_layer)
        ])
        if use_se:
            _layer.extend([
                SEModule(channels=dw_channels, reduction=se_ratio),
                layers.BatchNormalization(),
                layers.Dropout(drop_layer)
            ])
        _layer.append(act_layer)
        _layer.extend([layers.Conv2D(channels, kernel_size=1, strides=1),
                       layers.BatchNormalization(),
                       layers.Dropout(drop_layer)])

        self.conv = tf.keras.Sequential(_layer)

    def call(self, inputs, *args, **kwargs):
        out = self.conv(inputs)
        if self.use_shortcut:
            out += inputs
        return out


if __name__ == '__main__':
    x = tf.random.uniform(shape=[8, 24, 24, 16], minval=0, maxval=1)
    y = ReXBlock(in_channels=16, channels=8, t=1, stride=2, use_se=True, se_ratio=12, act_layer=None, drop_layer=None)(
        x)

    print(y.shape)
