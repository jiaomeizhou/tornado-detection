from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import logging
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.data.constants import CHANNEL_MIN_MAX, ALL_VARIABLES

def build_model(shape: Tuple[int] = (120, 240, 2),
                c_shape: Tuple[int] = (120, 240, 2),
                input_variables: List[str] = ALL_VARIABLES,
                start_filters: int = 64,
                l2_reg: float = 0.001,
                background_flag: float = -3.0,
                include_range_folded: bool = True,
                head='maxpool'):
    # Create input layers for each input_variable
    inputs = {}
    for v in input_variables:
        inputs[v] = tf.keras.Input(shape=shape, name=v)
    n_sweeps = shape[2]

    # Normalize inputs and concatenate along channel dimension
    normalized_inputs = tf.keras.layers.Concatenate(axis=-1, name='Concatenate1')(
        [normalize(inputs[v], v) for v in input_variables]
    )

    # Replace nan pixels with background flag
    normalized_inputs = FillNaNs(background_flag)(normalized_inputs)

    # Add channel for range folded gates
    if include_range_folded:
        range_folded = tf.keras.Input(shape=shape[:2] + (n_sweeps,), name='range_folded_mask')
        inputs['range_folded_mask'] = range_folded
        normalized_inputs = tf.keras.layers.Concatenate(axis=-1, name='Concatenate2')(
            [normalized_inputs, range_folded]
        )

    # Input coordinate information
    cin = tf.keras.Input(c_shape, name='coordinates')
    inputs['coordinates'] = cin

    x, c = normalized_inputs, cin

    # Add a Conv2D layer to change the number of channels to 3
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same', name='channel_adjust')(x)

    # Create the MobileNetV2 model
    base_model = MobileNetV2(
        include_top=False,  # Exclude the final classification layer
        input_shape=(shape[0], shape[1], 3),
        weights=None,  # No pre-trained weights
    )

    x = base_model(x)

    if head == 'mlp':
        # MLP head
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=4096, activation='relu')(x)
        x = tf.keras.layers.Dense(units=2024, activation='relu')(x)
        output = tf.keras.layers.Dense(1)(x)
    elif head == 'maxpool':
        # Per gridcell
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=1,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                   activation='relu', name='conv_512')(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=1,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                   activation='relu', name='conv_256')(x)
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, name='heatmap')(x)
        # Max in scene
        output = tf.keras.layers.GlobalMaxPooling2D()(x)

    return tf.keras.Model(inputs=inputs, outputs=output)

def normalize(x, name: str):
    """
    Channel-wise normalization using known CHANNEL_MIN_MAX
    """
    min_max = np.array(CHANNEL_MIN_MAX[name])  # [2,]
    n_sweeps = x.shape[-1]

    # Choose mean,var to get approximate [-1,1] scaling
    var = ((min_max[1] - min_max[0]) / 2) ** 2  # scalar
    var = np.array(n_sweeps * [var, ])  # [n_sweeps,]

    offset = (min_max[0] + min_max[1]) / 2  # scalar
    offset = np.array(n_sweeps * [offset, ])  # [n_sweeps,]

    return tf.keras.layers.Normalization(mean=offset,
                                         variance=var,
                                         name='Normalize_%s' % name)(x)