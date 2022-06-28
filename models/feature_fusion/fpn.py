# -*- coding = utf-8 -*-
# @Time : 2021/12/15 4:49 PM
# @Author : Yuri Su
from backbone.Darknet.darknet import Conv2D_BN_ReLU, CSPLayer
import tensorflow as tf
from tensorflow.keras import layers


def fpn(x, num_blocks):
    """
    feat1:64,64,256
    feat2:32,32,512
    feat3:16,16,1024
    Args:
        x:
        num_blocks:数组：csp次数
    Returns:

    """
    in_channels = [128, 256, 512, 1024]
    feat_1, feat_2, feat_3 = x
    P5 = Conv2D_BN_ReLU(feat_3, in_channels[2], (1, 1))
    P5_upsample = layers.UpSampling2D()(P5)  # 32,32,512
    P5_upsample = tf.concat([P5_upsample, feat_2], axis=-1)  # 32,32,1024
    P5_upsample = CSPLayer(P5_upsample, in_channels[1], num_blocks=num_blocks[0], name="backbone.ff.csp_1")  # 32,32,512

    P4 = Conv2D_BN_ReLU(P5_upsample, in_channels[1], (1, 1))  # 32,32,256
    P4_upsample = layers.UpSampling2D()(P4)  # 64,64,256
    P4_upsample = tf.concat([feat_1, P4_upsample], axis=-1)  # 64,64,512
    P4_upsample = CSPLayer(P4_upsample, in_channels[1], num_blocks=num_blocks[1], name="backbone.ff.csp_2")  # 64,64,256

    x = Conv2D_BN_ReLU(P4_upsample, in_channels[0], (1, 1))
    x = layers.UpSampling2D()(x)  # 128,128,128
    x = Conv2D_BN_ReLU(x, 64, (1, 1))
    return x
