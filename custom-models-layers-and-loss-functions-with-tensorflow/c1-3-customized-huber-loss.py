import keras
import tensorflow as tf
import numpy as np
from keras.losses import Loss

# inputs
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# labels
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=500, verbose=0)

print(model.predict([10.0]))


class HuberLoss(Loss):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)


model.compile(optimizer='sgd', loss=HuberLoss(1.02))
model.fit(xs, ys, epochs=500, verbose=0)

print(model.predict([10.0]))
