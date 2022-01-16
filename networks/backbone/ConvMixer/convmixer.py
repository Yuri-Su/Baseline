from tensorflow.keras import layers, Sequential, Model
import tensorflow as tf


class ConvMixLayer(layers.Layer):
    def __init__(self, dim, kernal_size=9):
        super().__init__()
        self.resnet = Sequential([
            layers.Conv2D(dim, kernel_size=kernal_size, groups=dim, padding='same'),
            layers.Activation("gelu"),
            layers.BatchNormalization()
        ])
        self.conv1x1 = Sequential([
            layers.Conv2D(dim, kernel_size=1),
            layers.Activation("gelu"),
            layers.BatchNormalization()
        ])

    def call(self, inputs, *args, **kwargs):
        x = inputs + self.resnet(inputs)
        x = self.conv1x1(x)
        return x


class ConvMixer(Model):
    def __init__(self, dim, depth, kernal_size=9, patch_size=7, n_classes=1000):
        super(ConvMixer, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(filters=dim, kernel_size=patch_size, strides=patch_size),
            layers.Activation("gelu"),
            layers.BatchNormalization()
        ])

        self.blocks = Sequential([ConvMixLayer(dim=dim, kernal_size=kernal_size) for _ in tf.range(depth)])

        self.head = Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Flatten(),
            layers.Dense(n_classes)
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs)
        x = self.blocks(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = ConvMixer(dim=512, depth=4)
    model.build(input_shape=(None, 224, 224, 3))
    print(model.summary())
