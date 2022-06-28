# -*- coding = utf-8 -*-
# @Time : 2022/1/5 9:02 PM
# @Author : Yuri Su
import tensorflow as tf
from tensorflow.keras import Sequential, layers
import tensorflow_addons as tfa

from networks.backbone.SEModule.SE_Module import SEModule
from networks.backbone.vision_transformer.vit import TransformerBlock


def conv3x3_bn(out_channels, downsample=False):
    stride = 1 if downsample is False else 2
    return Sequential([
        layers.Conv2D(filters=out_channels, strides=stride, kernel_size=(3, 3), use_bias=False, padding='same'),
        layers.BatchNormalization(),
        tfa.layers.GELU()
    ])


class PreNorm(layers.Layer):
    def __init__(self, fn, norm):
        super(PreNorm, self).__init__()
        self.norm = norm
        self.fn = fn

    def call(self, inputs, *args, **kwargs):
        return self.fn(self.norm(inputs), **kwargs)


class MBConv(layers.Layer):
    def __init__(self, in_channels, out_channels, downsample=False, expansion=4):
        super(MBConv, self).__init__()
        self.downsample = downsample
        stride = 1 if downsample is False else 2
        hidden_dim = int(in_channels * expansion)

        if self.downsample:
            self.pool = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
            self.proj = layers.Conv2D(filters=out_channels, kernel_size=1, strides=1, padding='VALID', use_bias=False)

        if expansion == 1:
            self.conv = Sequential([
                layers.Conv2D(hidden_dim, kernel_size=3, strides=stride, padding='same', groups=hidden_dim,
                              use_bias=False),
                layers.BatchNormalization(),
                tfa.layers.GELU(),
                layers.Conv2D(out_channels, kernel_size=1, strides=1, padding='valid', use_bias=False),
                layers.BatchNormalization()
            ])
        else:
            self.conv = Sequential([
                layers.Conv2D(hidden_dim, kernel_size=1, strides=stride, padding='valid', use_bias=False),
                layers.BatchNormalization(),
                tfa.layers.GELU(),
                SEModule(hidden_dim, reduction=4),
                layers.Conv2D(out_channels, kernel_size=1, strides=1, padding='valid', use_bias=False),
                layers.BatchNormalization()
            ])
        self.conv = PreNorm(self.conv, layers.BatchNormalization())

    def call(self, inputs, *args, **kwargs):
        if self.downsample:
            return self.proj(self.pool(inputs)) + self.conv(inputs)
        else:
            return inputs + self.conv(inputs)


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        h, w, c = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        return tf.reshape(output, shape=[batch_size, h, w, c])


class TransformerBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, expansion=4, heads=8, downsample=False, dropout=0.):
        super(TransformerBlock, self).__init__()
        hidden_dim = int(in_channels * expansion)
        self.downsample = downsample
        if self.downsample:
            self.pool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
            self.proj = layers.Conv2D(filters=out_channels, kernel_size=1, strides=1, use_bias=False)
        self.att = MultiHeadSelfAttention(embed_dim=out_channels, num_heads=heads)
        self.ffn = tf.keras.Sequential(
            [
                layers.LayerNormalization(),
                layers.Conv2D(hidden_dim, kernel_size=1, activation='gelu', use_bias=False),
                layers.Conv2D(out_channels, kernel_size=1, use_bias=False),
                layers.Dropout(dropout)
            ]
        )

    def call(self, inputs, *args, **kwargs):
        if self.downsample:
            shortcut = self.proj(self.pool(inputs))
        else:
            shortcut = inputs

        attn_output = self.att(shortcut)
        # attn_output = self.dropout(attn_output)
        out1 = attn_output + shortcut
        out2 = self.ffn(out1)
        print(f"out1.shape = {out1.shape} out2.shape = {out2.shape}")
        return out2 + out1


if __name__ == '__main__':
    x = tf.random.uniform(shape=[8, 32, 32, 16], minval=0, maxval=1)
    model = TransformerBlock(16, 32, downsample=True)
    y = model(x)
    print(y.shape)
