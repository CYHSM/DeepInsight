"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
from wavelets import WaveletAnalysis
import numpy as np


def wavelet_transform(signal, sampling_rate, average_window=1000, scaling_factor=0.25, wave_highpass=2, wave_lowpass=30000):
    """
    Calculates the wavelet transform for each point in signal, then averages
    each window and returns together fourier frequencies

    Parameters
    ----------
    signal : (N,1) array_like
        Signal to be transformed
    sampling_rate : int
        Sampling rate of signal
    average_window : int, optional
        Average window to downsample wavelet transformed input, by default 1000
    scaling_factor : float, optional
        Determines amount of log-spaced frequencies M in output, by default 0.25
    wave_highpass : int, optional
        Cut of frequencies below, by default 2
    wave_lowpass : int, optional
        Cut of frequencies above, by default 30000

    Returns
    -------
    wavelet_power : (N, M) array_like
        Wavelet transformed signal
    wavelet_frequencies : (M, 1) array_like
        Corresponding frequencies to wavelet_power
    """
    (wavelet_power, wavelet_frequencies, wavelet_obj) = simple_wavelet_transform(signal, sampling_rate,
                                                                                 scaling_factor=scaling_factor, wave_highpass=wave_highpass, wave_lowpass=wave_lowpass)

    # Average over window
    if average_window is not 1:
        wavelet_power = np.reshape(
            wavelet_power, (wavelet_power.shape[0], wavelet_power.shape[1] // average_window, average_window))
        wavelet_power = np.mean(wavelet_power, axis=2).transpose()
    else:
        wavelet_power = wavelet_power.transpose()

    return wavelet_power, wavelet_frequencies


def simple_wavelet_transform(signal, sampling_rate, scaling_factor=0.25, wave_lowpass=None, wave_highpass=None):
    """
    Simple wavelet transformation of signal

    Parameters
    ----------
    signal : (N,1) array_like
        Signal to be transformed
    sampling_rate : int
        Sampling rate of signal
    scaling_factor : float, optional
        Determines amount of log-space frequencies M in output, by default 0.25
    wave_highpass : int, optional
        Cut of frequencies below, by default 2
    wave_lowpass : int, optional
        Cut of frequencies above, by default 30000

    Returns
    -------
    wavelet_power : (N, M) array_like
        Wavelet transformed signal
    wavelet_frequencies : (M, 1) array_like
        Corresponding frequencies to wavelet_power
    wavelet_obj : object
        WaveletTransform Object
    """
    wavelet_obj = WaveletAnalysis(signal, dt=1 / sampling_rate, dj=scaling_factor)
    wavelet_power = wavelet_obj.wavelet_power
    wavelet_frequencies = wavelet_obj.fourier_frequencies

    if wave_lowpass or wave_highpass:
        wavelet_power = wavelet_power[(wavelet_frequencies < wave_lowpass) & (wavelet_frequencies > wave_highpass), :]
        wavelet_frequencies = wavelet_frequencies[(wavelet_frequencies < wave_lowpass) & (wavelet_frequencies > wave_highpass)]

    return (wavelet_power, wavelet_frequencies, wavelet_obj)
