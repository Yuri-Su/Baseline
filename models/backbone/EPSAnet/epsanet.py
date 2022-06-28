"""
Description:
Author: Yuri Su
Date: 2021-11-09 14:39:07
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from backbone.SEModule.SE_Module import SEModule


# Total params: 20,592,395
# Trainable params: 20,539,275
# Non-trainable params: 53,120
class PSAModule(layers.Layer):
    def __init__(self,
                 filters,
                 kernal_size=None,
                 groups=None,
                 stride=1,
                 ):
        super(PSAModule, self).__init__()
        if kernal_size is None:
            kernal_size = [3, 5, 7, 9]
        if groups is None:
            groups = [1, 4, 8, 16]
        self.conv_1 = layers.Conv2D(filters=filters // 4,
                                    kernel_size=kernal_size[0],
                                    padding='same',
                                    strides=stride,
                                    groups=groups[0])
        self.conv_2 = layers.Conv2D(filters=filters // 4,
                                    kernel_size=kernal_size[1],
                                    padding='same',
                                    strides=stride,
                                    groups=groups[1])
        self.conv_3 = layers.Conv2D(filters=filters // 4,
                                    kernel_size=kernal_size[2],
                                    padding='same',
                                    strides=stride,
                                    groups=groups[2])
        self.conv_4 = layers.Conv2D(filters=filters // 4,
                                    kernel_size=kernal_size[3],
                                    padding='same',
                                    strides=stride,
                                    groups=groups[3])
        self.se = SEModule(filters // 4)
        self.split_channel = filters // 4
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs, **kwargs):
        global out
        batch_size = tf.shape(inputs)[0]
        x1 = self.conv_1(inputs)
        x2 = self.conv_2(inputs)
        x3 = self.conv_3(inputs)
        x4 = self.conv_4(inputs)
        feats = tf.concat([x1, x2, x3, x4], axis=-1)

        feats = tf.reshape(feats, (batch_size, 4,
                                   tf.shape(feats)[1], tf.shape(feats)[2], self.split_channel))
        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)
        x_se = tf.concat((x1_se, x2_se, x3_se, x4_se), axis=-1)
        attention_vectors = tf.reshape(x_se, (batch_size, 4, 1,
                                              1, self.split_channel))

        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors

        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = tf.concat((x_se_weight_fp, out), -1)
        return out


'''
feature map大小不变，通道数扩大四倍
input_shape:[batch_size,w,h,channel]
output_shape:[batch_size,w,h,channelx4]
'''


class EPSABlock(layers.Layer):
    expansion = 4

    def __init__(self,
                 filters,
                 conv_kernels=None,
                 stride=1,
                 groups=None,
                 downsample=None,
                 norm_layer=None):
        super(EPSABlock, self).__init__()
        if groups is None:
            groups = [1, 4, 8, 16]
        if conv_kernels is None:
            conv_kernels = [3, 5, 7, 9]
        if norm_layer is None:
            norm_layer = layers.BatchNormalization
        self.conv1 = layers.Conv2D(filters, kernel_size=1)
        self.bn1 = norm_layer()
        self.conv2 = PSAModule(filters,
                               stride=stride, kernal_size=conv_kernels,
                               groups=groups)
        self.bn2 = norm_layer()
        self.conv3 = layers.Conv2D(filters * self.expansion, kernel_size=1)
        self.bn3 = norm_layer()
        self.relu = layers.ReLU()
        self.downsample = downsample

    def call(self, inputs, *args, **kwargs):
        identity = inputs
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(inputs)
        out += identity
        out = self.relu(out)
        return out


class EPSANet(layers.Layer):
    def __init__(self, num_layers):
        super(EPSANet, self).__init__()
        self.in_channels = 64
        self.conv1 = layers.Conv2D(self.in_channels, kernel_size=7, strides=2, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.layer1 = self._makelayer(64, num_layers[0], stride=1)
        self.layer2 = self._makelayer(128, num_layers[1], stride=2)
        self.layer3 = self._makelayer(256, num_layers[2], stride=2)
        self.layer4 = self._makelayer(512, num_layers[3], stride=2)

    def _makelayer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * EPSABlock.expansion:
            downsample = Sequential([
                layers.Conv2D(out_channels * EPSABlock.expansion, kernel_size=1, strides=stride),
                layers.BatchNormalization()
            ])
        layer = [EPSABlock(out_channels, stride=stride, downsample=downsample)]
        self.inplanes = out_channels * EPSABlock.expansion
        for i in range(1, num_blocks):
            layer.append(EPSABlock(out_channels))
        return tf.keras.Sequential(layer)

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feat1 = self.layer2(x)
        feat2 = self.layer3(feat1)
        feat3 = self.layer4(feat2)
        return feat1, feat2, feat3


def epsanet50():
    model = EPSANet([3, 4, 6, 3])
    return model


def epsanet101():
    model = EPSANet([3, 4, 23, 3])
    return model


if __name__ == '__main__':
    # x = tf.random.uniform(shape=[8, 224, 224, 3], minval=0, maxval=1)
    x = layers.Input([512, 512, 3])
    model = epsanet50()
    y1, y2, y3 = model(x)

    print(y1.shape, y2.shape, y3.shape)
