"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
import numpy as np


def calculate_speed_from_position(positions, interval, smoothing=False):
    """
    Calculate speed from X,Y coordinates

    Parameters
    ----------
    positions : (N, 2) array_like
        N samples of observations, containing X and Y coordinates
    interval : int
        Duration between observations (in s, equal to 1 / sr)
    smoothing : bool or int, optional 
        If speeds should be smoothed, by default False/0

    Returns
    -------
    speed : (N, 1) array_like
        Instantenous speed of the animal
    """
    X, Y = positions[:, 0], positions[:, 1]
    # Smooth diffs instead of speeds directly
    Xdiff = np.diff(X)
    Ydiff = np.diff(Y)
    if smoothing:
        Xdiff = smooth_signal(Xdiff, smoothing)
        Ydiff = smooth_signal(Ydiff, smoothing)
    speed = np.sqrt(Xdiff**2 + Ydiff**2) / interval
    speed = np.append(speed, speed[-1])

    return speed


def calculate_heading_direction_from_position(positions, smoothing=False, return_as_deg=False):
    """
    Calculates heading direction based on X and Y coordinates. With one measurement we can only calculate heading direction

    Parameters
    ----------
    positions : (N, 2) array_like
        N samples of observations, containing X and Y coordinates
    smoothing : bool or int, optional 
        If speeds should be smoothed, by default False/0
    return_as_deg : bool
        Return heading in radians or degree

    Returns
    -------
    heading_direction : (N, 1) array_like
        Heading direction of the animal
    """
    X, Y = positions[:, 0], positions[:, 1]
    # Smooth diffs instead of speeds directly
    Xdiff = np.diff(X)
    Ydiff = np.diff(Y)
    if smoothing:
        Xdiff = smooth_signal(Xdiff, smoothing)
        Ydiff = smooth_signal(Ydiff, smoothing)
    # Calculate heading direction
    heading_direction = np.arctan2(Ydiff, Xdiff)
    heading_direction = np.append(heading_direction, heading_direction[-1])
    if return_as_deg:
        heading_direction = heading_direction * (180 / np.pi)

    return heading_direction


def calculate_head_direction_from_leds(positions, return_as_deg=False):
    """
    Calculates head direction based on X and Y coordinates with two LEDs. 

    Parameters
    ----------
    positions : (N, 2) array_like
        N samples of observations, containing X and Y coordinates
    return_as_deg : bool
        Return heading in radians or degree

    Returns
    -------
    head_direction : (N, 1) array_like
        Head direction of the animal
    """
    X_led1, Y_led1, X_led2, Y_led2 = positions[:, 0], positions[:, 1], positions[:, 2], positions[:, 3]
    # Calculate head direction
    head_direction = np.arctan2(X_led1 - X_led2, Y_led1 - Y_led2)
    # Put in right perspective in relation to the environment
    offset = +np.pi/2
    head_direction = (head_direction + offset + np.pi) % (2*np.pi) - np.pi
    head_direction *= -1
    if return_as_deg:
        head_direction = head_direction * (180 / np.pi)

    return head_direction


def smooth_signal(signal, N):
    """
    Simple smoothing by convolving a filter with 1/N.

    Parameters
    ----------
    signal : array_like
        Signal to be smoothed
    N : int
        smoothing_factor

    Returns
    -------
    signal : array_like
            Smoothed signal
    """
    # Preprocess edges
    signal = np.concatenate([signal[0:N], signal, signal[-N:]])
    # Convolve
    signal = np.convolve(signal, np.ones((N,))/N, mode='same')
    # Postprocess edges
    signal = signal[N:-N]

    return signal
