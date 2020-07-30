import time
import os
import h5py
import deepinsight

import numpy as np
import unittest
unittest.TestLoader.sortTestMethodsUsing = None


class TestDeepInsight(unittest.TestCase):
    """Simple Testing Class"""

    def tearDown(self):
        time.sleep(0.1)

    def setUp(self):
        unittest.TestCase.setUp(self)
        np.random.seed(0)
        self.fp_deepinsight_folder = os.getcwd() + '/tests/test_files/'
        self.fp_deepinsight = self.fp_deepinsight_folder + 'test.h5'
        if os.path.exists(self.fp_deepinsight):
            os.remove(self.fp_deepinsight)
        else:
            os.makedirs(self.fp_deepinsight_folder)
        self.input_length = int(3e5)
        self.input_channels = 5
        self.sampling_rate = 30000
        self.input_output_ratio = 100

        self.rand_input = np.sin(np.random.rand(
            int(self.input_length), self.input_channels))
        self.rand_input_timesteps = np.arange(0, self.input_length)
        self.rand_output = np.random.rand(
            self.input_length // self.input_output_ratio)
        self.rand_timesteps = np.arange(
            0, self.input_length, self.input_output_ratio)

    def test_fullrun(self):
        """
        Tests wavelet transformation of random signal
        """
        # Transform raw data to frequency domain
        deepinsight.preprocess.preprocess_input(
            self.fp_deepinsight, self.rand_input, sampling_rate=self.sampling_rate)
        hdf5_file = h5py.File(self.fp_deepinsight, mode='r')
        # Get wavelets from hdf5 file
        input_wavelets = hdf5_file['inputs/wavelets']
        # Check statistics of wavelets
        np.testing.assert_almost_equal(np.mean(input_wavelets), 0.048329726)
        np.testing.assert_almost_equal(np.std(input_wavelets), 0.032383125)
        np.testing.assert_almost_equal(np.median(input_wavelets), 0.04608967)
        np.testing.assert_almost_equal(np.max(input_wavelets), 0.40853173)
        np.testing.assert_almost_equal(np.min(input_wavelets), 1.6544704e-05)
        hdf5_file.close()

        # Prepare outputs
        deepinsight.preprocess.preprocess_output(
            self.fp_deepinsight, self.rand_input_timesteps, self.rand_output, self.rand_timesteps)

        # Define loss functions and train model
        loss_functions = {'aligned': 'mse'}
        loss_weights = {'aligned': 1}
        user_opts = {'epochs': 2, 'steps_per_epoch': 10,
                     'validation_steps': 10, 'log_output': False, 'save_model': True}

        deepinsight.train.run_from_path(
            self.fp_deepinsight, loss_functions, loss_weights, user_opts)

        # Get loss and shuffled loss for influence plot, both is also stored back to HDF5 file
        losses, output_predictions, indices = deepinsight.analyse.get_model_loss(
            self.fp_deepinsight, stepsize=10)

        shuffled_losses = deepinsight.analyse.get_shuffled_model_loss(
            self.fp_deepinsight, axis=1, stepsize=10)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
