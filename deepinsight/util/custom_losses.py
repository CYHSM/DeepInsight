"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def euclidean_loss(y_true, y_pred):
    # We use tf.sqrt instead of K.sqrt as there is a bug in K.sqrt (as of March 14, 2018)
    res = tf.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    return res


def cyclical_mae_rad(y_true, y_pred):
    return K.mean(K.minimum(K.abs(y_pred - y_true), K.minimum(K.abs(y_pred - y_true + 2*np.pi), K.abs(y_pred - y_true - 2*np.pi))), axis=-1)


def mse(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


def mae(y_true, y_pred):
    return tf.keras.losses.MAE(y_true, y_pred)
