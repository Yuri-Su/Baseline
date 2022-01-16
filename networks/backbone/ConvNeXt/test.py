import tensorflow as tf

from networks.backbone.ConvNeXt.build import build_model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

model = build_model(name='convnext_tiny', num_classes=10)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=10)