"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


def plot_residuals(fp_hdf_out, output_names, aggregator=np.mean, frequency_spacing=1):
    """
    Plots influence plots for each output

    Parameters
    ----------
    fp_hdf_out : str
        File path to HDF5 file
    aggregator : function handle, optional
        Which aggregator to use for plotting the lineplots, by default np.mean
    frequency_spacing : int, optional
        Spacing on x axis between frequencies, by default 1
    """
    # Read data from HDF5 file
    hdf5_file = h5py.File(fp_hdf_out, mode='r')
    losses = hdf5_file["analysis/losses"][()]
    shuffled_losses = hdf5_file["analysis/influence/shuffled_losses"][()]
    frequencies = hdf5_file["inputs/fourier_frequencies"][()].astype(np.float32)
    hdf5_file.close()

    # Calculate residuals, make sure there is no division by zero by adding small constant. TODO Should be relative to loss
    residuals = (shuffled_losses - losses) / (losses + 0.1)

    # Plot
    fig, axes = plt.subplots(1, len(output_names), figsize=(16, 5))
    if len(output_names) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for all_residuals, ax, on in zip(residuals.transpose(), axes, output_names):
        residuals_mean = np.mean(all_residuals, axis=0)
        all_residuals = all_residuals / np.sum(residuals_mean)
        df_to_plot = pd.DataFrame(all_residuals).melt()
        sns.lineplot(x="variable", y="value", data=df_to_plot, ax=ax, estimator=aggregator, ci=68, marker='o',
                     color='k').set(xlabel='Frequencies (Hz)', ylabel='Frequency Influence (%)')
        ax.set_xticks(np.arange(0, len(frequencies), frequency_spacing))
        ax.set_xticklabels(np.round(frequencies[0::frequency_spacing], 2), fontsize=8, rotation=45)
        ax.set_title(on)
    for ax in axes:
        ax.invert_xaxis()
    sns.despine()
    fig.tight_layout()
    fig.show()
