import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer


class DefenseLayer(Layer):
    def __init__(self, **kwargs):
        super(DefenseLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        super(DefenseLayer, self).build(input_shape)  

    def call(self, x):
        return keras.backend.pow(x-x,1/3)

    def compute_output_shape(self, input_shape):
        return (None,input_shape[1],input_shape[2],input_shape[3])


