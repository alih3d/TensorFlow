import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

x_s = np.random.randint(-30,50,10)
y_s = 32 + 1.8 * x_s

#  Sequential Model
# model = tf.keras.Sequential([
#
# Dense(units=1, use_bias=True, input_shape=[1])
#
# ])

#  model using Functional API
inputs = tf.keras.Input(shape=(1,))
outputs =  Dense(10, use_bias=True,)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1),
              loss='mean_squared_error',
              metrics=['mse'])
model.fit(x=x_s, y=y_s, epochs=500, verbose=0)


pred = model.predict([10])

print(pred)