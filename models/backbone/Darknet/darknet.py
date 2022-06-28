# -*- coding = utf-8 -*-
# @Time : 2021/10/26 2:38 下午
# @Author : Yuri Su


import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import UpSampling2D, Activation, LeakyReLU
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Layer, MaxPooling2D)
from backbone.MSFF.multi_scale_feature_fusion import MSFF


class Focus(Layer):
    def __init__(self):
        super(Focus, self).__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2 if input_shape[1] is not None else input_shape[1],
                input_shape[2] // 2 if input_shape[2] is not None else input_shape[2], input_shape[3] * 4)

    def call(self, x, **kwargs):
        return tf.concat(
            [x[..., ::2, ::2, :],
             x[..., 1::2, ::2, :],
             x[..., ::2, 1::2, :],
             x[..., 1::2, 1::2, :]],
            axis=-1
        )


def Conv2D_BN_ReLU(x, filters, kernal_size, stride=1, padding='same', name=''):
    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=stride, padding=padding)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def SPPBottleneck(x, out_channels, name=""):
    x = Conv2D_BN_ReLU(x, out_channels // 2, kernal_size=(1, 1), name=name + '.conv1')
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = Conv2D_BN_ReLU(x, out_channels, kernal_size=(1, 1), name=name + '.conv2')
    return x


def Bottleneck(x, out_channels, shortcut=True, name=""):
    y = Conv2D_BN_ReLU(x, out_channels, (1, 1), name=name + '.conv1')
    y = Conv2D_BN_ReLU(x, out_channels, (3, 3), name=name + '.conv2')
    if shortcut:
        y = Add()([x, y])
    return y


def CSPLayer(x, num_filters, num_blocks, shortcut=True, expansion=0.5, name=""):
    hidden_channels = int(num_filters * expansion)
    x_1 = Conv2D_BN_ReLU(x, hidden_channels, kernal_size=(1, 1), name=name + '.conv1')
    x_2 = Conv2D_BN_ReLU(x, hidden_channels, (1, 1), name=name + '.conv2')
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, name=name + '.m.' + str(i))
    route = Concatenate()([x_1, x_2])
    return Conv2D_BN_ReLU(route, num_filters, (1, 1), name=name + '.conv3')


def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, name=""):
    x = Conv2D_BN_ReLU(x, num_filters, kernal_size=(3, 3), stride=2, name=name + '.0')
    if last:
        x = SPPBottleneck(x, num_filters, name=name + '.1')
    return CSPLayer(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion,
                    name=name + '.1' if not last else name + '.2')


def darknet_body(x):
    base_channels = 64  # 64
    base_depth = 3  # 3
    # 512, 512, 3 => 256, 256, 12
    x = Focus()(x)
    # 256, 256, 12 => 256, 256, 64
    x = Conv2D_BN_ReLU(x, base_channels, (3, 3), name='backbone.backbone.stem.conv')
    # print("shape:", x.shape)
    # x = EPSABlock(base_channels)(x)
    # 256, 256, 64 => 128, 128, 128
    x = resblock_body(x, base_channels * 2, base_depth, name='backbone.backbone.dark2')
    # 128, 128, 128 => 64, 64, 256
    x = resblock_body(x, base_channels * 4, base_depth * 3, name='backbone.backbone.dark3')
    feat1 = x
    # 64, 64, 256 => 32, 32, 512
    x = resblock_body(x, base_channels * 8, base_depth * 3, name='backbone.backbone.dark4')
    feat2 = x
    # 32, 32, 512 => 16, 16, 1024
    x = resblock_body(x, base_channels * 16, base_depth, shortcut=False, last=True, name='backbone.backbone.dark5')
    feat3 = x
    return feat1, feat2, feat3


def centernet_csp_head(x, num_classes):
    # feat1:64,64,256
    # feat2:32,32,512
    # feat3:16,16,1024
    in_channels = [256, 512, 1024]
    feat1, feat2, feat3 = x
    num_filters = 256
    # -------------------------------#
    #   解码器
    # -------------------------------#

    P5 = Conv2D_BN_ReLU(feat3, in_channels[1], (1, 1), name='backbone.lateral_conv0')
    P5_upsample = UpSampling2D()(P5)  # 32,32,512
    P5_upsample = MSFF(channels=in_channels[1], r=4)(feat2, P5_upsample)  # 32,32,512

    P4 = Conv2D_BN_ReLU(P5_upsample, in_channels[0], (1, 1), name='backbone.lateral_conv1')  # 32,32,256
    P4_upsample = UpSampling2D()(P4)  # 64,64,256
    P4_upsample = MSFF(in_channels[0], 4)(feat1, P4_upsample)  # 64,64,256
    x = UpSampling2D()(P4_upsample)  # 128,128,256
    x = CSPLayer(x, 128, 1, name='backbone.last')
    # x = DarknetConv2D_BN_SiLU(128, (1, 1), name='backbone.lateral_conv2')(x)
    # 16, 16, 2048  ->  32, 32, 256 -> 64, 64, 128 -> 128, 128, 64
    # for i in range(3):
    #     # 进行上采样
    #     x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
    #                         kernel_initializer='he_normal',
    #                         kernel_regularizer=l2(5e-4))(x)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    # 最终获得128,128,64的特征层

    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False)(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(num_classes, 1,
                activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv2D(2, 1, )(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(2, 1)(y3)
    return y1, y2, y3


if __name__ == '__main__':
    # x = tf.random.uniform(shape=[8, 512, 512, 3], minval=0, maxval=1)
    x = layers.Input([512, 512, 3])
    y1, y2, y3 = darknet_body(x)
    print(y1.shape, y2.shape, y3.shape)
    model = tf.keras.Model(x, [y1, y2, y3])
    print(model.summary())
