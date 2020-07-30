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
        time.sleep(1)

    def setUp(self):
        unittest.TestCase.setUp(self)
        np.random.seed(0)
        self.fp_deepinsight = 'tests/test_files/test.h5'
        if os.path.exists(self.fp_deepinsight):
            os.remove(self.fp_deepinsight)
        else:
            os.mkdir(self.fp_deepinsight, parents=True, exist_ok=True)
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

    def test01_input_output(self):
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

    def test02_preprocessing(self):
        # Prepare outputs
        deepinsight.preprocess.preprocess_output(
            self.fp_deepinsight, self.rand_input_timesteps, self.rand_output, self.rand_timesteps)

    def test03_model_training(self):
        # Define loss functions and train model
        loss_functions = {'output_aligned': 'mse'}
        loss_weights = {'output_aligned': 1}
        user_opts = {'epochs': 2, 'steps_per_epoch': 10,
                     'validation_steps': 10, 'log_output': False, 'save_model': True}

        deepinsight.train.run_from_path(
            self.fp_deepinsight, loss_functions, loss_weights, user_opts)

    def test04_model_performance(self):
        # Get loss and shuffled loss for influence plot, both is also stored back to HDF5 file
        losses, output_predictions, indices = deepinsight.analyse.get_model_loss(
            self.fp_deepinsight, stepsize=10)

        # Test cases
        np.testing.assert_almost_equal(losses[-1], 1.0168755e-05)
        np.testing.assert_almost_equal(losses[0], 0.53577816)
        np.testing.assert_almost_equal(np.mean(losses), 0.09069238)
        np.testing.assert_almost_equal(np.std(losses), 0.13594063)
        np.testing.assert_almost_equal(np.median(losses), 0.045781307)
        np.testing.assert_almost_equal(np.max(losses), 0.53577816)
        np.testing.assert_almost_equal(np.min(losses), 1.0168755e-05)

    def test05_model_shuffling(self):
        shuffled_losses = deepinsight.analyse.get_shuffled_model_loss(
            self.fp_deepinsight, axis=1, stepsize=10)

        # Test cases
        np.testing.assert_almost_equal(np.mean(shuffled_losses), 0.09304095)
        np.testing.assert_almost_equal(np.std(shuffled_losses), 0.13982493)
        np.testing.assert_almost_equal(np.median(shuffled_losses), 0.04165206)
        np.testing.assert_almost_equal(np.max(shuffled_losses), 0.7405345)
        np.testing.assert_almost_equal(np.min(shuffled_losses), 2.0834877e-07)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
