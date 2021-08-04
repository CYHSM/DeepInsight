"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence

from . import hdf5


def create_train_and_test_generators(opts):
    """
    Creates training and test generators given opts dictionary

    Parameters
    ----------
    opts : dict
        Dictionary holding options for data creation and model training

    Returns
    -------
    training_generator : object
        Sequence class used for generating training data
    testing_generator : object
        Sequence class used for generating testing data
    """
    # 1.) Create training generator
    training_generator = RawWaveletSequence(opts, training=True)
    # 2.) Create testing generator
    testing_generator = RawWaveletSequence(opts, training=False)
    # 3.) Assert that training and testing data are different

    return (training_generator, testing_generator)


class RawWaveletSequence(Sequence):
    """
    Data Generator class. Import functions are get_input_sample and get_output_sample. 
    Each call to __getitem__ will yield a (input, output) pair

    Parameters
    ----------
    Sequence : object
        Keras sequence

    Yields
    -------
    input_sample : array_like
        Batched input for model training
    output_sample : array_like
        Batched output for model optimization
    """

    def __init__(self, opts, training):
        # 1.) Set all options as attributes
        self.set_opts_as_attribute(opts)

        # 2.) Load data memmaped for mean/std estimation and fast plotting
        self.wavelets = hdf5.read_hdf_memmapped(self.fp_hdf_out, 'inputs/wavelets')

        # Get output(s)
        outputs = []
        for key, value in opts['loss_functions'].items():
            tmp_out = hdf5.read_hdf_memmapped(self.fp_hdf_out, 'outputs/' + key)
            outputs.append(tmp_out)
        self.outputs = outputs

        # 3.) Prepare for training
        self.training = training
        self.prepare_data_generator(training=training)

    def __len__(self):
        return len(self.cv_indices)

    def __getitem__(self, idx):
        # 1.) Define start and end index
        if self.shuffle:
            idx = np.random.choice(self.cv_indices)
        else:
            idx = self.cv_indices[idx]
        cut_range = np.arange(idx, idx + self.sample_size)

        # 2.) Above takes consecutive batches, implement random batching here
        if self.random_batches:
            indices = np.random.choice(self.cv_indices, size=self.batch_size)
            cut_range = [np.arange(start_index, start_index + self.model_timesteps) for start_index in indices]
            cut_range = np.array(cut_range)
        else:
            cut_range = np.reshape(cut_range, (self.batch_size, cut_range.shape[0] // self.batch_size))

        # 3.) Get input sample
        input_sample = self.get_input_sample(cut_range)

        # 4.) Get output sample
        output_sample = self.get_output_sample(cut_range)

        return (input_sample, output_sample)

    def get_input_sample(self, cut_range):
        # 1.) Cut Ephys / fancy indexing for memmap is planned, if fixed use: cut_data = self.wavelets[cut_range, self.fourier_frequencies, self.channels]
        cut_data = self.wavelets[cut_range, :, :]
        cut_data = np.reshape(cut_data, (cut_data.shape[0] * cut_data.shape[1], cut_data.shape[2], cut_data.shape[3]))

        # 2.) Normalize input
        cut_data = (cut_data - self.est_mean) / self.est_std

        # 3.) Reshape for model input
        cut_data = np.reshape(cut_data, (self.batch_size, self.model_timesteps, cut_data.shape[1], cut_data.shape[2]))

        # 4.) Take care of optional settings
        cut_data = np.transpose(cut_data, axes=(0, 3, 1, 2))
        cut_data = cut_data[..., np.newaxis]

        return cut_data

    def get_output_sample(self, cut_range):
        # 1.) Cut Ephys
        out_sample = []
        for out in self.outputs:
            cut_data = out[cut_range, ...]
            cut_data = np.reshape(cut_data, (cut_data.shape[0] * cut_data.shape[1], cut_data.shape[2]))

            # 2.) Reshape for model output
            if len(cut_data.shape) is not self.batch_size:
                cut_data = np.reshape(cut_data, (self.batch_size, self.model_timesteps, cut_data.shape[1]))

            # 3.) Divide evenly and make sure last output is being decoded
            if self.average_output:
                cut_data = cut_data[:, np.arange(0, cut_data.shape[1] + 1, self.average_output)[1::] - 1]
            out_sample.append(cut_data)

        return out_sample

    def prepare_data_generator(self, training):
        # 1.) Define sample size and means
        self.sample_size = self.model_timesteps * self.batch_size

        if training:
            self.cv_indices = self.training_indices
        else:
            self.cv_indices = self.testing_indices
            
        # Make sure random choice takes from array not list 500x speedup
        self.cv_indices = np.array(self.cv_indices)
            
        # 9.) Calculate normalization for wavelets
        # Debug
        print(self.fp_hdf_out)
        print(os.path.dirname(self.fp_hdf_out))
        print(os.path.basename(self.fp_hdf_out))
        print(os.path.basename(self.fp_hdf_out)[:-3])
        print(self.training_indices)
        print(self.testing_indices)
        meanstd_path = os.path.dirname(self.fp_hdf_out) + '/models/tmp/' + os.path.basename(self.fp_hdf_out)[:-3] + '_meanstd_start{}_end{}_tstart{}_tend{}.p'.format(
            self.training_indices[0], self.training_indices[-1], self.testing_indices[0], self.testing_indices[-1])
        
        if os.path.exists(meanstd_path):
            (self.est_mean, self.est_std) = pickle.load(open(meanstd_path, 'rb'))
        else:
            print('Calculating MAD normalization parameters')
            if len(self.training_indices) > 1e5:
                print('Downsampling wavelets for MAD calculation')
                self.est_mean = np.median(self.wavelets[self.training_indices[::100], :, :], axis=0)
                self.est_std = np.median(abs(self.wavelets[self.training_indices[::100], :, :] - self.est_mean), axis=0)                
            else:
                self.est_mean = np.median(self.wavelets[self.training_indices, :, :], axis=0)
                self.est_std = np.median(abs(self.wavelets[self.training_indices, :, :] - self.est_mean), axis=0)
            pickle.dump((self.est_mean, self.est_std), open(meanstd_path, 'wb'))

        # Make sure indices contain no NaN values
        if self.handle_nan:
            self.cv_indices = self.check_for_nan()
            
        # 10.) Define output shape. Most robust way is to get a dummy input and take that shape as output shape
        (dummy_input, dummy_output) = self.__getitem__(0)
        # Corresponds to the output of this generator, aka input to model. Also remove batch shape,
        self.input_shape = dummy_input.shape[1:]

    def set_opts_as_attribute(self, opts):
        for k, v in opts.items():
            setattr(self, k, v)

    def get_name(self):
        name = ""
        for attr in self.important_attributes:
            name += attr + ':{},'.format(getattr(self, attr))
        return name[:-1]
    
    def check_for_nan(self):
        new_cv_indices, len_before = [], len(self.cv_indices)
        for idx, cv in enumerate(self.cv_indices):
            if not idx % 100000:
                print('{} / {}'.format(cv, self.cv_indices[-1]), end='\r')
            cut_range = np.arange(cv, cv + self.sample_size)
            cut_range = np.reshape(cut_range, (self.batch_size, cut_range.shape[0] // self.batch_size))
            out_sample = self.get_output_sample(cut_range)
            nan_in_out = any([any(np.isnan(x.flatten())) for x in out_sample[0]])
            if not nan_in_out:
                new_cv_indices.append(cv)
        print('Len before {}, after {} --- Diff {}'.format(len_before, len(new_cv_indices), len_before - len(new_cv_indices)))
        return np.array(new_cv_indices)
