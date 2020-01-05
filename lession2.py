from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print("tensorflow version: {}".format(tf.__version__))

celsius_q = np.array([   -40, -10,  0,  8, 15, 22,  38])
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100])

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished the model training")

"""
import matplotlib.pyplot as plt
plt.xlabel("Epoch number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
"""

print("Predict with 100 and 15: {}".format(model.predict([100.0, 15.0])))
print("These are the layer variables: {}".format(l0.get_weights()))
