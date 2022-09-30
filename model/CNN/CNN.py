
from tensorflow import keras
import tensorflow as tf
# ignore error
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def cnn(input_shape):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            tf.keras.layers.Conv1D(filters=128, kernel_size=20, strides=3, padding='same', activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool1D(pool_size=2, strides=3),
            tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
            tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu),

            tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax'),
        ]
    )
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.03), metrics=['accuracy'])
    return model
