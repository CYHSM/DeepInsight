"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
import os
import numpy as np
import h5py

from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

import keras.backend as K

from . import architecture
from . import util


def train_model_on_generator(model, training_generator, testing_generator, loss_functions, loss_weights, steps_per_epoch=300, validation_steps=300,
                             epochs=10, tensorboard_logfolder='./', model_name='', verbose=1, reduce_lr=False, log_output=False, save_model_only=False, compile_only=False):
    """
    Function for training a given model, with data provided by training and testing generators

    Parameters
    ----------
    model : object
        Keras model
    training_generator : object
        Data generator for training data
    testing_generator : object
        Data generator for testing data
    loss_functions : dict
        Selected loss function for each behaviour
    loss_weights : dict
        Selected weights for each loss function
    steps_per_epoch : int, optional
        Number of steps for training the model, by default 300
    validation_steps : int, optional
        Number of steps for validating the model, by default 300
    epochs : int, optional
        Number of epochs to train model, by default 10
    tensorboard_logfolder : str, optional
        Where to store tensorboard logfiles, by default './'
    model_name : str, optional
        Name of selected model, used to return best model, by default ''
    verbose : int, optional
        Verbosity level, by default 1
    reduce_lr : bool, optional
        If True reduce learning rate on plateau, by default False
    log_output : bool, optional
        Log the output to tensorflow logfolder, by default False
    save_model_only : bool, optional
        Save best model after each epoch, by default False
    compile_only : bool, optional
        If true returns only compiled model, by default False

    Returns
    -------
    model : object
        Keras model
    history : dict
        Dictionary containing training and validation performance
    """
    # Compile model
    opt = optimizers.Adam(lr=training_generator.learning_rate, amsgrad=True)
    # Check if there are multiple outputs
    for key, item in loss_functions.items():
        try:
            function_handle = getattr(util.custom_losses, item)
        except AttributeError:
            function_handle = item
        loss_functions[key] = function_handle
    model.compile(loss=loss_functions, optimizer=opt, loss_weights=loss_weights)
    if compile_only:  # What a hack. Keras bug from Oct9 in saving/loading models.
        return model
    # Get model name for storing tmp files
    if model_name is '':
        model_name = training_generator.get_name()
    # Initiate callbacks
    callbacks = []
    if reduce_lr:
        reduce_lr_cp = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
        callbacks.append(reduce_lr_cp)
    if log_output:
        tensorboard_cp = TensorBoard(log_dir=tensorboard_logfolder)
        callbacks.append(tensorboard_cp)
    if save_model_only:
        file_name = model_name + '.hdf5'
        model_cp = ModelCheckpoint(filepath=file_name, save_best_only=True, save_weights_only=True)
        callbacks.append(model_cp)
    # Run model training
    try:
        history = model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, shuffle=training_generator.shuffle,
                                      validation_steps=validation_steps, validation_data=testing_generator, verbose=verbose, callbacks=callbacks)
    except KeyboardInterrupt:
        print('-> Notebook interrupted')
        history = []
    finally:
        if save_model_only:  # Make sure interruption of jupyter notebook returns best model
            model.load_weights(file_name)
            print('-> Returning best Model')
    return (model, history)


def train_model(model_path, path_in, tensorboard_logfolder, model_tmp_path, loss_functions, loss_weights, num_cvs=5):
    """
    Trains the model across the experiment using cross validation and saves the model files
    TODO Save models back to HDF5 to keep everything in one place

    Parameters
    ----------
    model_path : str
        Path to where model should be stored
    path_in : str
        Path to HDF5 File
    tensorboard_logfolder : str
        Path to where tensorboard logs should be stored
    model_tmp_path : str
        Temporary file path used for returning best fit model
    loss_functions : dict
        For each output the corresponding loss function
    loss_weights : dict
        For each output the corresponding weight
    num_cvs : int, optional
        Number of cross validation splits, by default 5
    """
    # Get experiment length
    hdf5_file = h5py.File(path_in, mode='r')
    tmp_wavelets = hdf5_file['inputs/wavelets']
    tmp_opts = util.opts.get_opts(path_in, train_test_times=(np.array([]), np.array([])))
    exp_indices = np.arange(0, tmp_wavelets.shape[0] - tmp_opts['model_timesteps'])
    cv_splits = np.array_split(exp_indices, num_cvs)
    for cv_run, cvs in enumerate(cv_splits):
        K.clear_session()
        # For cv
        training_indices = np.setdiff1d(exp_indices, cvs)  # All except the test indices
        testing_indices = cvs
        # opts -> generators -> model
        opts = util.opts.get_opts(path_in, train_test_times=(training_indices, testing_indices))
        opts['loss_functions'] = loss_functions.copy()
        opts['loss_weights'] = loss_weights
        opts['loss_names'] = list(loss_functions.keys())
        opts['num_cvs'] = num_cvs
        (training_generator, testing_generator) = util.data_generator.create_train_and_test_generators(opts)
        model = get_model_from_function(training_generator, show_summary=True)

        print('------------------------------------------------')
        print('-> Model and generators loaded')
        print('------------------------------------------------')

        (model, history) = train_model_on_generator(model, training_generator, testing_generator, loss_functions=loss_functions.copy(), loss_weights=loss_weights, reduce_lr=True, log_output=True,
                                                    tensorboard_logfolder=tensorboard_logfolder, model_name=model_tmp_path, save_model_only=True,
                                                    steps_per_epoch=opts['steps_per_epoch'], validation_steps=opts['validation_steps'], epochs=opts['epochs'])
        # Save model and history
        if history:
            opts['history'] = history.history
        cv_model_path = model_path[0:-3] + '_' + str(cv_run) + '.h5'
        util.hdf5.save_model_with_opts(model, opts, cv_model_path)
        print('------------------------------------------------')
        print('-> Model_{} saved to {}'.format(cv_run, cv_model_path))
        print('------------------------------------------------')
    hdf5_file.close()


def run_from_path(path_in, loss_functions, loss_weights):
    """
    Runs model training giving path to HDF5 file and loss dictionaries

    Parameters
    ----------
    path_in : str
        Path to HDF5
    loss_functions : dict
        For each output the corresponding loss function
    loss_weights : dict
        For each output the corresponding weight
    """
    dirname = os.path.dirname(path_in)
    filename = os.path.basename(path_in)
    # Define folders
    tensorboard_logfolder = dirname + '/logs/' + filename[0:-3]  # Remove .h5 for logfolder
    model_tmp_path = dirname + '/models/tmp/tmp_model'
    model_path = dirname + '/models/' + filename[0:-3] + '_model.h5'
    # Create folders if needed
    for f in [os.path.dirname(model_tmp_path), os.path.dirname(model_path)]:
        if not os.path.exists(f):
            os.makedirs(f)
    print('------------------------------------------------')
    print('-> Running {} from {}'.format(filename, dirname))
    print('- Logs : {} \n- Model temporary : {} \n- Model : {}'.format(tensorboard_logfolder, model_tmp_path, model_path))
    print('------------------------------------------------')
    # Train model
    print('------------------------------------------------')
    print('Starting standard model')
    print('------------------------------------------------')
    train_model(model_path, path_in, tensorboard_logfolder, model_tmp_path, loss_functions, loss_weights)


def get_model_from_function(training_generator, show_summary=True):
    model_function = getattr(architecture, training_generator.model_function)
    model = model_function(training_generator, show_summary=show_summary)

    return model
