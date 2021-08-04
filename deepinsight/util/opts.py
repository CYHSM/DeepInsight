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
    opts['fp_hdf_out'] = fp_hdf_out  # Filepath for hdf5 file storing wavelets and outputs
    opts['sampling_rate'] = 30  # Sampling rate of the wavelets
    opts['training_indices'] = train_test_times[0].tolist()  # Indices into wavelets used for training the model, adjusted during CV
    opts['testing_indices'] = train_test_times[1].tolist()  # Indices into wavelets used for testing the model, adjusted during CV

    # -------- MODEL PARAMETERS --------------
    opts['model_function'] = 'the_decoder'  # Model architecture used
    opts['model_timesteps'] = 64  # How many timesteps are used in the input layer, e.g. a sampling rate of 30 will yield 2.13s windows. Has to be divisible X times by 2. X='num_convs_tsr'
    opts['num_convs_tsr'] = 4  # Number of downsampling steps within the model, e.g. with model_timesteps=64, it will downsample 64->32->16->8->4 and output 4 timesteps
    opts['average_output'] = 2**opts['num_convs_tsr']  # Whats the ratio between input and output shape
    opts['channel_lower_limit'] = 2

    opts['optimizer'] = 'adam'  # Learning algorithm
    opts['learning_rate'] = 0.0007  # Learning rate
    opts['kernel_size'] = 3  # Kernel size for all convolutional layers
    opts['conv_padding'] = 'same'  # Which padding should be used for the convolutional layers
    opts['act_conv'] = 'elu'  # Activation function for convolutional layers
    opts['act_fc'] = 'elu'  # Activation function for fully connected layers
    opts['dropout_ratio'] = 0  # Dropout ratio for fully connected layers
    opts['filter_size'] = 64  # Number of filters in convolutional layers
    opts['num_units_dense'] = 1024  # Number of units in fully connected layer
    opts['num_dense'] = 2  # Number of fully connected layers
    opts['gaussian_noise'] = 1  # How much gaussian noise is added (unit = standard deviation)

    # -------- TRAINING----------------------
    opts['batch_size'] = 8  # Batch size used for training the model
    opts['steps_per_epoch'] = 250  # Number of steps per training epoch
    opts['validation_steps'] = 250  # Number of steps per validation epoch
    opts['epochs'] = 20  # Number of epochs
    opts['shuffle'] = True  # If input should be shuffled
    opts['random_batches'] = True  # If random batches in time are used
    opts['metrics'] = []
    opts['last_layer_activation_function'] = 'linear'
    opts['handle_nan'] = False

    # -------- MISC--------------- ------------
    opts['tensorboard_logfolder'] = './'  # Logfolder for tensorboard
    opts['model_folder'] = './'  # Folder for saving the model
    opts['log_output'] = False  # If output should be logged
    opts['save_model'] = False  # If model should be saved

    return opts
