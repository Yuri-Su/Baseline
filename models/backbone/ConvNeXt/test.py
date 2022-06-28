import tensorflow as tf

from networks.backbone.ConvNeXt.convnext import create_model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

model = create_model(model_name='convnext_tiny_224', input_shape=(32, 32), num_classes=10, include_top=True,
                     pretrained=True, use_tpu=False)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=5)
