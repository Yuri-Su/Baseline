# -*- coding = utf-8 -*-
# @Time : 2021/12/15 4:35 PM
# @Author : Yuri Su
from tensorflow.python.keras.layers import BatchNormalization, Conv2DTranspose, Activation
from tensorflow.python.keras.regularizers import l2


def upsample_n(x, num_filters, num):
    """
    16, 16, 2048  ->  32, 32, 256 -> 64, 64, 128 -> 128, 128, 64
    Args:
        x:
        num_filters: 输出filters = num_filters//pow(2,num)
        num:

    Returns:

    """

    for i in range(num):
        # 进行上采样
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x
