"""
This module implements masked loss functions for handling NaN values in the target data.
These loss functions can be used in TensorFlow/Keras models to ignore NaN values during training
and evaluation.

.. note::
    
    Be very careful about precision, since here ``tf.reduce_sum(mask)``
    can be very big if the number of points in a batch is large.
    Here we use ``float32`` globally to avoid issues, 
    that ensure that we don't get overflow in the summation 
    when the number of points is less than :math:`3.4 \cdot 10^{38}`.
    Using mixed precision with ``float16`` would lead to overflow
    when the number of points in batch is larger than :math:`6 \cdot 10^{4}`.
"""

import tensorflow as tf
from keras.utils import register_keras_serializable


@register_keras_serializable()
def masked_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    .. admonition:: Description

        Compute the masked mean squared error (MSE) between the true and predicted values,
        ignoring NaN values in the true data.
    
    :param y_true: The ground truth values.
    :param y_pred: The predicted values.

    :returns:
        - loss (``tf.Tensor``) -- The computed masked mean squared error.
    """
    mask = tf.cast(~tf.math.is_nan(y_true), y_pred.dtype)
    y_true = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_sum(mask * tf.square(y_pred - y_true)) / tf.reduce_sum(mask)

@register_keras_serializable()
def masked_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    .. admonition:: Description

        Compute the masked mean absolute error (MAE) between the true and predicted values,
        ignoring NaN values in the true data.
    
    :param y_true: The ground truth values.
    :param y_pred: The predicted values.

    :returns:
        - loss (``tf.Tensor``) -- The computed masked mean absolute error.
    """
    mask = tf.cast(~tf.math.is_nan(y_true), y_pred.dtype)
    y_true = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_sum(mask * tf.abs(y_pred - y_true)) / tf.reduce_sum(mask)