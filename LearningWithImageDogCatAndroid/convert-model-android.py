#!/usr/bin/env python3

import tensorflow as tf

# Create a converter that will read the Keras model file. The
# input_shapes parameter is necessary, since apparently the model file
# does not contain this information. The value here is an array with
# values for batch size, two input image dimensions, and the number of
# color layers. The name of the layer, input_1, I picked up from the
# error message that I got when trying to run without setting
# input_shapes.
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('mnist_shift.h5')

# Run the conversion and write to a new file.
tflite_model = converter.convert()
with open('mnist_shift.tflite', 'wb') as f:
    f.write(tflite_model)
