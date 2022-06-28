# -*- coding = utf-8 -*-
# @Time : 2022/1/2 3:12 PM
# @Author : Yuri Su
import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
    Conv2D
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


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
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

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
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, expansion=4, dropout=0.1,downsample=True):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                LayerNormalization(),
                Conv2D(embed_dim * expansion, kernel_size=1, activation='gelu', use_bias=False),
                Conv2D(embed_dim, kernel_size=1, use_bias=False),
                Dropout(dropout)
            ]
        )

    def call(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout(attn_output)
        out1 = attn_output + inputs
        out2 = self.mlp(out1)
        return out2 + out1


class VisionTransformer(tf.keras.Model):
    def __init__(
            self,
            image_size,
            patch_size,
            num_layers,
            num_classes,
            d_model,
            num_heads,
            mlp_dim,
            channels=3,
            dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers
        # 缩放
        self.rescale = Rescaling(1.0 / 255)
        # 位置编码，可训练参数
        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches + 1, d_model)
        )
        # 类别编码，可训练参数
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        self.patch_proj = Dense(d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                LayerNormalization(epsilon=1e-6),
                Dense(mlp_dim, activation=tf.nn.gelu),
                Dropout(dropout),
                Dense(num_classes),
            ]
        )

    # 原图片切片,最后展平
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        # 归一化
        x = self.rescale(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        # 类别编码和输入堆叠
        x = tf.concat([class_emb, x], axis=1)
        # 位置编码和x相加
        x = x + self.pos_emb
        # 堆叠 encoder 块
        for layer in self.enc_layers:
            x = layer(x, training)
        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x


if __name__ == '__main__':
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=4,
        num_classes=10,
        d_model=64,
        num_heads=4,
        mlp_dim=128,
        channels=3,
        dropout=0.1,
    )
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        optimizer=tf.optimizers.SGD(),
        metrics=["accuracy"],
    )
    history = model.fit(x_train, y_train, epochs=8, batch_size=32)
    print(history)
    # x = tf.random.uniform(shape=[8, 32, 32, 16], minval=0, maxval=1)
    # model = MultiHeadSelfAttention(embed_dim=16, num_heads=8)
    # y = model(x)
    # print(y.shape)
