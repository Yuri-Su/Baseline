from networks.backbone.ConvNeXt.build import build_model
import tensorflow as tf


def main():
    model = build_model(name="convnext_tiny")
    x = tf.random.uniform(shape=(8, 224, 224, 3), minval=0, maxval=1)
    y = model(x)
    print(model.summary())
    print(y.shape)


if __name__ == '__main__':
    main()
