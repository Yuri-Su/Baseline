from platform import machine
from keras import activations
from keras import layers
from pip import main


def ReLU():
    return layers.ReLU()


def ReLU6():
    return activations.relu6


def Sigmoid():
    return activations.sigmoid


def LeakyReLU(alpha=0.3):
    return layers.LeakyReLU(alpha=alpha)


def Tanh():
    return activations.tanh


class HSigmoid(layers.Layer):

    def __init__(self, bias=3.0, divisor=6.0, min_value=0.0, max_value=1.0):
        super(HSigmoid, self).__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def call(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)


def HSwish():
    return activations.hard_sigmoid


class Swish(layers.Layer):

    def __init__(self):
        super(Swish, self).__init__()

    def call(self, inputs):
        return inputs * activations.sigmoid(inputs)


def GELU():
    return activations.gelu