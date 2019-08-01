# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


# install tensorflow 2.0

# !pip install tensorflow==2.0.0-beta1


import tensorflow as tf

print(tf.__version__)


mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train:[batch, 28, 28], y_train: [labels]

print (len(x_train),  x_train[0][0])

print (len(y_train), y_train)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)

print ("evaluating .... ")

model.evaluate(x_test, y_test, batch_size=64)