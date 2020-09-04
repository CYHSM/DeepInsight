"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
from . import util
import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tensorflow.compat.v1.keras import backend as K
import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def get_model_loss(fp_hdf_out, stepsize=1, shuffles=None, axis=0, verbose=1):
    """
    Loops across cross validated models and calculates loss and predictions for full experiment length

    Parameters
    ----------
    fp_hdf_out : str
        File path to HDF5 file
    stepsize : int, optional
        Determines how many samples will be evaluated. 1 -> N samples evaluated, 
        2 -> N/2 samples evaluated, etc..., by default 1
    shuffles : dict, optional
        If wavelets should be shuffled, important for calculating influence scores, by default None

    Returns
    -------
    losses : (N,1) array_like
        Loss between predicted and ground truth observation
    predictions : dict
        Dictionary with predictions for each behaviour, each item in dict has size (N, Z) with Z the dimensions of the sample (e.g. Z_position=2, Z_speed=1, ...)
    indices : (N,1) array_like
        Indices which were evaluated, important when taking stepsize unequal to 1
    """
    dirname = os.path.dirname(fp_hdf_out)
    filename = os.path.basename(fp_hdf_out)[0:-3]
    cv_results = []
    (_, _, _, opts) = util.hdf5.load_model_with_opts(
        dirname + '/models/' + filename + '_model_{}.h5'.format(0))
    loss_names = opts['loss_names']
    time_shift = opts['model_timesteps']
    if verbose > 0:
        progress_bar = tf.keras.utils.Progbar(
            opts['num_cvs'], width=30, verbose=1, interval=0.05, unit_name='run')
    for k in range(0, opts['num_cvs']):
        K.clear_session()
        # Find folders
        model_path = dirname + '/models/' + filename + '_model_{}.h5'.format(k)
        # Load model and generators
        (model, training_generator, testing_generator,
         opts) = util.hdf5.load_model_with_opts(model_path)
        # -----------------------------------------------------------------------------------------------
        if shuffles is not None:
            testing_generator = shuffle_wavelets(
                training_generator, testing_generator, shuffles)
        losses, predictions, indices = calculate_losses_from_generator(
            testing_generator, model, verbose=0, stepsize=stepsize)
        # -----------------------------------------------------------------------------------------------
        cv_results.append((losses, predictions, indices))
        if verbose > 0:
            progress_bar.add(1)
    cv_results = np.array(cv_results)
    # Reshape cv_results
    losses = np.concatenate(cv_results[:, 0], axis=0)
    predictions = {k: [] for k in loss_names}
    for out in cv_results[:, 1]:
        for p, name in zip(out, loss_names):
            predictions[name].append(p)
    for key, item in predictions.items():
        if stepsize > 1:
            tmp_output = np.concatenate(predictions[key], axis=0)[:, -1, :]
        else:
            tmp_output = np.concatenate(predictions[key], axis=0)[:, -1, :]
            tmp_output = np.array([np.pad(l, [time_shift, 0], mode='constant', constant_values=[l[0], 0])
                                   for l in tmp_output.transpose()]).transpose()
        predictions[key] = tmp_output
    indices = np.concatenate(cv_results[:, 2], axis=0)
    # We only take the last timestep for decoding, so decoder does not see any part of the future
    indices = indices + time_shift
    if stepsize > 1:
        losses = losses[:, :, -1]
    else:
        losses = losses[:, :, -1]
        losses = np.array([np.pad(l, [time_shift, 0], mode='constant', constant_values=[l[0], 0])
                           for l in losses.transpose()]).transpose()
        indices = np.arange(0, losses.shape[0])
    # Also save to HDF5
    hdf5_file = h5py.File(fp_hdf_out, mode='a')
    for key, item in predictions.items():
        util.hdf5.create_or_update(hdf5_file, dataset_name="analysis/predictions/{}_axis{}_stepsize{}".format(key, axis, stepsize),
                                   dataset_shape=item.shape, dataset_type=np.float32, dataset_value=item)
    util.hdf5.create_or_update(hdf5_file, dataset_name="analysis/losses_axis{}_stepsize{}".format(axis, stepsize),
                               dataset_shape=losses.shape, dataset_type=np.float32, dataset_value=losses)
    util.hdf5.create_or_update(hdf5_file, dataset_name="analysis/indices_axis{}_stepsize{}".format(axis, stepsize),
                               dataset_shape=indices.shape, dataset_type=np.int64, dataset_value=indices)
    hdf5_file.close()

    # Report model performance
    if verbose > 0:
        df_stats = calculate_model_stats(fp_hdf_out, losses, predictions, indices)
        print(df_stats)

    return losses, predictions, indices


def get_shuffled_model_loss(fp_hdf_out, stepsize=1, axis=0, verbose=1):
    """
    Shuffles the wavelets and recalculates error

    Parameters
    ----------
    fp_hdf_out : str
        File path to HDF5 file
    stepsize : int, optional
        Determines how many samples will be evaluated. 1 -> N samples evaluated, 
        2 -> N/2 samples evaluated, etc..., by default 1
    axis : int, optional
        Which axis to shuffle

    Returns
    -------
    shuffled_losses : (N,1) array_like
        Loss between predicted and ground truth observation for shuffled wavelets on specified axis
    """
    if axis == 0:
        raise ValueError(
            'Shuffling across time dimension (axis=0) not supported yet.')
    hdf5_file = h5py.File(fp_hdf_out, mode='r')
    tmp_wavelets_shape = hdf5_file['inputs/wavelets'].shape
    hdf5_file.close()
    shuffled_losses = []
    if verbose > 0:
        progress_bar = tf.keras.utils.Progbar(
            tmp_wavelets_shape[axis], width=30, verbose=1, interval=0.05, unit_name='run')
    for s in range(0, tmp_wavelets_shape[axis]):
        if axis == 1:
            losses, _, _ = get_model_loss(
                fp_hdf_out, stepsize=stepsize, shuffles={'f': s}, axis=axis, verbose=0)
        elif axis == 2:
            losses, _, _ = get_model_loss(
                fp_hdf_out, stepsize=stepsize, shuffles={'c': s}, axis=axis, verbose=0)
        shuffled_losses.append(losses)
        if verbose > 0:
            progress_bar.add(1)
    shuffled_losses = np.array(shuffled_losses)
    # Also save to HDF5
    hdf5_file = h5py.File(fp_hdf_out, mode='a')
    util.hdf5.create_or_update(hdf5_file, dataset_name="analysis/influence/shuffled_losses_axis{}_stepsize{}".format(axis, stepsize),
                               dataset_shape=shuffled_losses.shape, dataset_type=np.float32, dataset_value=shuffled_losses)
    hdf5_file.close()

    return shuffled_losses


def calculate_losses_from_generator(tg, model, num_steps=None, stepsize=1, verbose=0):
    """
    Keras evaluate_generator only returns a scalar loss (mean) while predict_generator only returns the predictions but not the real labels
    TODO Make it batch size independent

    Parameters
    ----------
    tg : object
        Data generator
    model : object
        Keras model
    num_steps : int, optional
        How many steps should be evaluated, by default None (runs through full experiment)
    stepsize : int, optional
        Determines how many samples will be evaluated. 1 -> N samples evaluated, 
        2 -> N/2 samples evaluated, etc..., by default 1
    verbose : int, optional
        Verbosity level

    Returns
    -------
    losses : (N,1) array_like
        Loss between predicted and ground truth observation
    predictions : dict
        Dictionary with predictions for each behaviour, each item in dict has size (N, Z) with Z the dimensions of the sample (e.g. Z_position=2, Z_speed=1, ...)
    indices : (N,1) array_like
        Indices which were evaluated, important when taking stepsize unequal to 1
    """
    # X.) Parse inputs
    if num_steps is None:
        num_steps = len(tg)

    # 1.) Make a copy and adjust attributes
    tmp_dict = tg.__dict__.copy()
    if tg.batch_size != 1:
        tg.batch_size = 1
        tg.random_batches = False
        tg.shuffle = False
        tg.sample_size = tg.model_timesteps * tg.batch_size

    # 2.) Get output tensors
    sess = K.get_session()
    (_, test_out) = tg.__getitem__(0)
    real_tensor, calc_tensors = K.placeholder(), []
    for output_index in range(0, len(test_out)):
        prediction_tensor = model.outputs[output_index]
        loss_tensor = model.loss_functions[output_index].fn(
            real_tensor, prediction_tensor)
        calc_tensors.append((prediction_tensor, loss_tensor))

    # 3.) Predict
    losses, predictions, indices = [], [], []
    for i in range(0, num_steps, stepsize):
        (in_tg, out_tg) = tg.__getitem__(i)
        indices.append(tg.cv_indices[i])
        loss, prediction = [], []
        for o in range(0, len(out_tg)):
            evaluated = sess.run(calc_tensors[o], feed_dict={
                                 model.input: in_tg, real_tensor: out_tg[o]})
            prediction.append(evaluated[0][0, ...])
            loss.append(evaluated[1][0, ...])  # Get rid of batch dimensions
        predictions.append(prediction)
        losses.append(loss)
        if verbose > 0 and not i % 50:
            print('{} / {}'.format(i, num_steps), end='\r')
    if verbose > 0:
        print('Performed {} gradient steps'.format(num_steps // stepsize))
    losses, predictions, indices = np.array(
        losses), swap_listaxes(predictions), np.array(indices)
    tg.__dict__.update(tmp_dict)

    return losses, predictions, indices


def shuffle_wavelets(training_generator, testing_generator, shuffles):
    """
    Shuffle procedure for model interpretation

    Parameters
    ----------
    training_generator : object
        Data generator for training data
    testing_generator : object
        Data generator for testing data
    shuffles : dict
        Indicates which axis to shuffle and which index in selected dimension, e.g. {'f' : 5} shuffles frequency axis 5

    Returns
    -------
    testing_generator : object
        Data generator for testing data with shuffled wavelets
    """
    rolled_wavelets = training_generator.wavelets.copy()
    for key, item in shuffles.items():
        if key == 'f':
            np.random.shuffle(rolled_wavelets[:, item, :])  # In place
        elif key == 'c':
            np.random.shuffle(rolled_wavelets[:, :, item])  # In place
        elif key == 't':
            np.random.shuffle(rolled_wavelets[item, :, :])  # In place
    testing_generator.wavelets = rolled_wavelets
    return testing_generator


def swap_listaxes(list_in):
    list_out = []
    for o in range(0, len(list_in[0])):
        list_out.append(np.array([out[o] for out in list_in]))
    return list_out


def calculate_model_stats(fp_hdf_out, losses, predictions, indices, additional_metrics=[spearmanr]):
    """
    Calculates statistics on model predictions

    Parameters
    ----------
    fp_hdf_out : str
        File path to HDF5 file
    losses : (N,1) array_like
        Loss between predicted and ground truth observation
    predictions : dict
        Dictionary with predictions for each behaviour, each item in dict has size (N, Z) with Z the dimensions of the sample (e.g. Z_position=2, Z_speed=1, ...)
    indices : (N,1) array_like
        Indices which were evaluated, important when taking stepsize unequal to 1
    additional_metrics : list, optional
        Additional metrics besides Pearson and Model loss to be evaluated, should take arguments (y_true, y_pred) and return scalar or first argument as metric

    Returns
    -------
    df_scores
        Dataframe of evaluated scores
    """
    hdf5_file = h5py.File(fp_hdf_out, mode='r')
    output_scores = []
    for idx, (key, y_pred) in enumerate(predictions.items()):
        y_true = hdf5_file['outputs/{}'.format(key)][indices, :]

        pearson_mean, additional_mean = 0, np.zeros((len(additional_metrics)))
        for p in range(y_pred.shape[1]):
            pearson_mean += np.corrcoef(y_true[:, p], y_pred[:, p])[0, 1]
            for add_idx, am in enumerate(additional_metrics):
                am_eval = am(y_true[:, p], y_pred[:, p])
                if len(am_eval) > 1:
                    am_eval = am_eval[0]
                additional_mean[add_idx] += am_eval
        additional_mean /= y_pred.shape[1]
        pearson_mean /= y_pred.shape[1]
        loss_mean = np.mean(losses[:, idx])
        output_scores.append((pearson_mean, loss_mean, *additional_mean))
    additional_columns = [f.__name__.title() for f in additional_metrics]
    df_scores = pd.DataFrame(output_scores, index=predictions.keys(), columns=['Pearson', 'Model Loss', *additional_columns])
    hdf5_file.close()

    return df_scores
