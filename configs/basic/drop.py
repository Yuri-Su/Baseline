import tensorflow as tf
from keras import layers


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs
    keep_prob = 1.0 - drop_prob

    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0], ) + (1, ) * (len(tf.shape(inputs)) - 1)
    random_tensor += (tf.random.uniform(shape, dtype=inputs.dtype))
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(layers.Layer):

    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_path

    def call(self, inputs, *args, **kwargs):
        return drop_path(inputs, self.drop_prob, self.training)
