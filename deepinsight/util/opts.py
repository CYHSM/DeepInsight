"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""


def get_opts(fp_hdf_out, train_test_times):
    """
    Returns the options dictionary which contains all parameters needed to
    create DataGenerator and train the model
    TODO Find better method of parameter storing (config files, store in HDF5, etc...)

    Parameters
    ----------
    fp_hdf_out : str
        File path to HDF5 file
    train_test_times : array_like
        Indices for training and testing generator

    Returns
    -------
    opts : dict
        Dictionary containing all model and training parameters
    """
    opts = dict()
    # -------- DATA ------------------------
    opts['fp_hdf_out'] = fp_hdf_out
    opts['sampling_rate'] = 30  # Sampling rate of the wavelets
    opts['training_indices'] = train_test_times[0].tolist()
    opts['testing_indices'] = train_test_times[1].tolist()

    # -------- MODEL PARAMETERS --------------
    opts['model_function'] = 'the_decoder'
    opts['model_timesteps'] = 64  # How many timesteps in the input layer

    opts['optimizer'] = 'adam'
    opts['learning_rate'] = 0.0007
    opts['kernel_size'] = 3
    opts['conv_padding'] = 'same'
    opts['act_conv'] = 'elu'
    opts['act_fc'] = 'elu'
    opts['dropout_ratio'] = 0
    opts['filter_size'] = 64
    opts['num_units_dense'] = 1024
    opts['num_dense'] = 2
    opts['gaussian_noise'] = 1
    opts['num_convs_tsr'] = 4
    opts['average_output'] = 2**opts['num_convs_tsr']

    # -------- TRAINING----------------------
    opts['batch_size'] = 8
    opts['steps_per_epoch'] = 250
    opts['validation_steps'] = 250
    opts['epochs'] = 20
    opts['shuffle'] = True
    opts['random_batches'] = True

    # -------- MISC--------------- ------------
    opts['tensorboard_logfolder'] = './'
    opts['model_folder'] = './'

    return opts
