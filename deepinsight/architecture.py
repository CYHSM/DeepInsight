"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
from tensorflow.keras.layers import Conv2D, GaussianNoise, TimeDistributed, Input, Dense, Lambda, Flatten, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def the_decoder(tg, show_summary=True):
    """
    Model architecture used for decoding from wavelet transformed neural signals

    Parameters
    ----------
    tg : object
        Data generator, holding all important options for creating and training the model
    show_summary : bool, optional
        Whether to show a summary of the model after creation, by default True

    Returns
    -------
    model : object
        Keras model
    """
    model_input = Input(shape=tg.input_shape)

    x = GaussianNoise(tg.gaussian_noise)(model_input)
    # timestep reductions
    for nct in range(0, tg.num_convs_tsr):
        x = TimeDistributed(Conv2D(filters=tg.filter_size, kernel_size=(tg.kernel_size, tg.kernel_size), strides=(
            2, 1), padding=tg.conv_padding, activation=tg.act_conv, name='conv_tsr{}'.format(nct)))(x)
        x = TimeDistributed(Conv2D(filters=tg.filter_size, kernel_size=(tg.kernel_size, tg.kernel_size), strides=(
            1, 2), padding=tg.conv_padding, activation=tg.act_conv, name='conv_fr{}'.format(nct)))(x)

    # batch x 128 x 60 x 11
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1, 4)))(x)

    layer_counter = 0
    while (K.int_shape(x)[3] > 2):
        x = TimeDistributed(Conv2D(filters=tg.filter_size * 2, kernel_size=(1, 2), strides=(1, 2),
                                   padding=tg.conv_padding, activation=tg.act_conv, name='conv_after_tsr{}'.format(layer_counter)))(x)
        layer_counter += 1

    # Flatten and fc
    x_flat = TimeDistributed(Flatten())(x)

    outputs = []
    for (key, item), output in zip(tg.loss_functions.items(), tg.outputs):
        x = x_flat
        for d in range(0, tg.num_dense):
            x = Dense(tg.num_units_dense, activation=tg.act_fc, name='dense{}_combine{}'.format(d, key))(x)
            x = Dropout(tg.dropout_ratio)(x)
        out = Dense(output.shape[1], name='{}'.format(key), activation='linear')(x)
        outputs.append(out)

    model = Model(inputs=model_input, outputs=outputs)

    if show_summary:
        print(model.summary(line_length=100))

    return model
