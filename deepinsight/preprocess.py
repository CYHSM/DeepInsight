"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
import time
from joblib import Parallel, delayed
import numpy as np
import h5py
import tensorflow as tf  # Progress bar only
import deepinsight.util.wavelet_transform as wt
from deepinsight.util import hdf5


def preprocess_input(fp_hdf_out, raw_data, average_window=1000, channels=None, window_size=100000,
                     gap_size=50000, sampling_rate=30000, scaling_factor=0.5, num_cores=4, **args):
    """
    Transforms raw neural data to frequency space, via wavelet transform implemented currently with aaren-wavelets (https://github.com/aaren/wavelets)
    Saves wavelet transformed data to HDF5 file (N, P, M) - (Number of timepoints, Number of frequencies, Number of channels)

    Parameters
    ----------
    fp_hdf_out : str
        File path to HDF5 file
    raw_data : (N, M) file or array_like
        Variable storing the raw_data (N data points, M channels), should allow indexing
    average_window : int, optional
        Average window to downsample wavelet transformed input, by default 1000
    channels : array_like, optional
        Which channels from raw_data to use, by default None
    window_size : int, optional
        Window size for calculating wavelet transformation, by default 100000
    gap_size : int, optional
        Gap size for calculating wavelet transformation, by default 50000
    sampling_rate : int, optional
        Sampling rate of raw_data, by default 30000
    scaling_factor : float, optional
        Determines amount of log-spaced frequencies P in output, by default 0.5
    num_cores : int, optional
        Number of paralell cores to use to calculate wavelet transformation, by default 4
    """
    # Get number of chunks
    if channels is None:
        channels = np.arange(0, raw_data.shape[1])
    num_points = raw_data.shape[0]
    if window_size > num_points:
        num_chunks = len(channels)
        output_size = raw_data.shape[0]
        mean_signal = np.mean(raw_data, axis=1)
        average_window = 1
        full_transform = True
    else:
        num_chunks = (num_points // gap_size) - 1
        output_size = ((num_chunks + 1) * gap_size) // average_window
        full_transform = False

    # Get estimate for number of frequencies
    (_, wavelet_frequencies) = wt.wavelet_transform(np.ones(window_size), sampling_rate, average_window, scaling_factor, **args)
    num_fourier_frequencies = len(wavelet_frequencies)
    # Prepare output file
    hdf5_file = h5py.File(fp_hdf_out, mode='a')
    if "inputs/wavelets" not in hdf5_file:
        hdf5_file.create_dataset("inputs/wavelets", [output_size, num_fourier_frequencies, len(channels)], np.float32)
        hdf5_file.create_dataset("inputs/fourier_frequencies", [num_fourier_frequencies], np.float16)
    # Makes saving 5 times faster as last index saving is fancy indexing and therefore slow
    hdf5_file.create_dataset("inputs/tmp_wavelets", [len(channels), output_size, num_fourier_frequencies], np.float32)

    # Prepare par pool
    par = Parallel(n_jobs=num_cores, verbose=0)

    # Start parallel wavelet transformation
    print('Starting wavelet transformation (n={}, chunks={}, frequencies={})'.format(
        num_points, num_chunks, num_fourier_frequencies))
    progress_bar = tf.keras.utils.Progbar(num_chunks, width=30, verbose=1, interval=0.05, unit_name='chunk')
    for c in range(0, num_chunks):
        if full_transform:
            raw_chunk = raw_data[:, c] - mean_signal
        else:
            start = gap_size * c
            end = start + window_size
            raw_chunk = raw_data[start: end, channels]
            # Process raw chunk
            raw_chunk = preprocess_chunk(raw_chunk, subtract_mean=True, convert_to_milivolt=False)

        # Calculate wavelet transform
        if full_transform:
            (wavelet_power, wavelet_frequencies) = wt.wavelet_transform(raw_chunk,
                                                                        sampling_rate=sampling_rate, scaling_factor=scaling_factor, average_window=average_window, **args)
        else:
            wavelet_transformed = np.zeros((raw_chunk.shape[0] // average_window, num_fourier_frequencies, raw_chunk.shape[1]))
            for ind, (wavelet_power, wavelet_frequencies) in enumerate(par(delayed(wt.wavelet_transform)(raw_chunk[:, i], sampling_rate, average_window, scaling_factor, **args) for i in range(0, raw_chunk.shape[1]))):
                wavelet_transformed[:, :, ind] = wavelet_power

        # Save in output file
        if full_transform:
            hdf5_file["inputs/tmp_wavelets"][c, :, :] = wavelet_power
        else:
            wavelet_index_end = end // average_window
            wavelet_index_start = start // average_window
            index_gap = gap_size // 2 // average_window
            if c == 0:
                this_index_start = 0
                this_index_end = wavelet_index_end - index_gap
                hdf5_file["inputs/wavelets"][this_index_start:this_index_end, :, :] = wavelet_transformed[0: -index_gap, :, :]
            elif c == num_chunks - 1:  # Make sure the last one fits fully
                this_index_start = wavelet_index_start + index_gap
                this_index_end = wavelet_index_end
                hdf5_file["inputs/wavelets"][this_index_start:this_index_end, :, :] = wavelet_transformed[index_gap::, :, :]
            else:
                this_index_start = wavelet_index_start + index_gap
                this_index_end = wavelet_index_end - index_gap
                hdf5_file["inputs/wavelets"][this_index_start:this_index_end, :, :] = wavelet_transformed[index_gap: -index_gap, :, :]
        hdf5_file.flush()
        progress_bar.add(1)

    # 7.) Put frequencies in and close file
    if full_transform:
        wavelet_power = np.transpose(hdf5_file["inputs/tmp_wavelets"], axes=(1, 2, 0))
        del hdf5_file["inputs/tmp_wavelets"]
        hdf5_file["inputs/wavelets"][:] = wavelet_power
    hdf5_file["inputs/fourier_frequencies"][:] = wavelet_frequencies
    hdf5_file.flush()
    hdf5_file.close()


def preprocess_chunk(raw_chunk, subtract_mean=True, convert_to_milivolt=False):
    """
    Preprocesses a chunk of data.

    Parameters
    ----------
    raw_chunk : array_like
        Chunk of raw_data to preprocess
    subtract_mean : bool, optional
        Subtract mean over all other channels, by default True
    convert_to_milivolt : bool, optional
        Convert chunk to milivolt , by default False

    Returns
    -------
    raw_chunk : array_like
        preprocessed_chunk
    """
    # Subtract mean across all channels
    if subtract_mean:
        raw_chunk = raw_chunk.transpose() - np.mean(raw_chunk.transpose(), axis=0)
        raw_chunk = raw_chunk.transpose()
    # Convert to milivolt
    if convert_to_milivolt:
        raw_chunk = raw_chunk * (0.195 / 1000)
    return raw_chunk


def preprocess_output(fp_hdf_out, raw_timestamps, output, output_timestamps, average_window=1000, dataset_name='aligned', dataset_type=np.float16):
    """
    Base file for preprocessing outputs (handles M-D case as of March2020).
    For more complex cases use specialized functions (see for example preprocess_output in util.tetrode module)

    Parameters
    ----------
    fp_hdf_out : str
        File path to HDF5 file
    raw_timestamps : (N,1) array_like
        Timestamps for each sample in continous
    output : array_like
        M dimensional output which will be aligned with continous
    output_timestamps : (N,1) array_like
        Timestamps for output
    average_window : int, optional
        Downsampling factor for raw data and output, by default 1000
    dataset_name : str, optional
        Field name for output stored in HDF5 file
    """
    hdf5_file = h5py.File(fp_hdf_out, mode='a')

    # Get size of wavelets
    input_length = hdf5_file['inputs/wavelets'].shape[0]

    # Get positions of both LEDs
    raw_timestamps = raw_timestamps[()]  # Slightly faster than np.array
    if output.ndim == 1:
        output = output[..., np.newaxis]

    output_aligned = np.array([np.interp(raw_timestamps[np.arange(0, raw_timestamps.shape[0],
                                                                  average_window)], output_timestamps, output[:, i]) for i in range(output.shape[1])]).transpose()

    # Create and save datasets in HDF5 File
    hdf5.create_or_update(hdf5_file, dataset_name="outputs/{}".format(dataset_name),
                          dataset_shape=[input_length, output_aligned.shape[1]], dataset_type=dataset_type, dataset_value=output_aligned[0: input_length, ...])
    hdf5_file.flush()
    hdf5_file.close()
    print('Successfully written Dataset="{}" to {}'.format(dataset_name, fp_hdf_out))
