from keras import layers


class SELayer(layers.Layer):

    def __init__(
            self,
            channels,
            squeeze_channels=None,
            ratio=16,
            act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
    ):
        super().__init__()
        assert len(act_cfg) == 2
        self.global_avgpool = layers.GlobalAveragePooling2D()
        if squeeze_channels is None:
            squeeze_channels = channels // ratio
        assert isinstance(squeeze_channels, int) and squeeze_channels > 0, \
            '"squeeze_channels" should be a positive integer, but get ' + \
            f'{squeeze_channels} instead.'

        self.conv1 = layers.Conv2D(filters=squeeze_channels,
                                   kernel_size=1,
                                   strides=1,
                                   padding='VALID',
                                   activation=act_cfg[0])
        self.conv2 = layers.Conv2D(filters=channels,
                                   kernel_size=1,
                                   strides=1,
                                   padding='VALID',
                                   activation=act_cfg[1])

    def call(self, inputs, *args, **kwargs):
        out = self.global_avgpool(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        return inputs * out