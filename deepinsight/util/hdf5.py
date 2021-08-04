"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
import h5py
import numpy as np
from . import data_generator


def create_or_update(hdf5_file, dataset_name, dataset_shape, dataset_type, dataset_value):
    """
    Create or update dataset in HDF5 file

    Parameters
    ----------
    hdf5_file : File
        File identifier
    dataset_name : str
        Name of new dataset
    dataset_shape : array_like
        Shape of new dataset
    dataset_type : type
        Type of dataset (np.float16, np.float32, 'S', etc...)
    dataset_value : array_like
        Data to store in HDF5 file
    """
    if not dataset_name in hdf5_file:
        hdf5_file.create_dataset(dataset_name, dataset_shape, dataset_type)
        hdf5_file[dataset_name][:] = dataset_value
    else:
        if hdf5_file[dataset_name].shape != dataset_shape:
            del hdf5_file[dataset_name]
            hdf5_file.create_dataset(dataset_name, dataset_shape, dataset_type)
        hdf5_file[dataset_name][:] = dataset_value
    hdf5_file.flush()


def save_model_with_opts(model, opts, file_name):
    """
    Saves Keras model and training options to HDF5 file
    Uses Keras save_weights for creating the model HDF5 file and then inserts into that

    Parameters
    ----------
    model : object
        Keras model
    opts : dict
        Dictionary used for training the model
    file_name : str
        Path to save to
    """
    model.save_weights(file_name)
    hdf5_file = h5py.File(file_name, mode='a')
    hdf5_file['opts'] = str(opts)
    hdf5_file.flush()
    hdf5_file.close()


def load_model_with_opts(file_name):
    """
    Load Keras model and training options from HDF5 file
    TODO: Remove eval and find better way of storing dict in HDF5 (hickle, pytables, etc...)

    Parameters
    ----------
    file_name : str
        Model path

    Returns
    -------
    model : object
        Keras model
    training_generator : object
        Datagenerator used to create training samples on the fly
    testing_generator : object
        Datagenerator used to create testing samples on the fly
    opts : dict
        Dictionary used for training the model
    """
    from .. import train
    # Get options from dictionary, stored as str in HDF5 (not recommended, TODO)
    hdf5_file = h5py.File(file_name, mode='r')
    opts = eval(hdf5_file['opts'][()])
    opts['handle_nan'] = False
    hdf5_file.close()

    # Use options to create data generators and model weights
    (training_generator, testing_generator) = data_generator.create_train_and_test_generators(opts)

    model = train.get_model_from_function(training_generator, show_summary=False)
    model = train.train_model_on_generator(model, training_generator, testing_generator,
                                           loss_functions=opts['loss_functions'], loss_weights=opts['loss_weights'], compile_only=True)
    model.load_weights(file_name)

    return (model, training_generator, testing_generator, opts)


def read_hdf_memmapped(fn_hdf, hdf_group):
    """
    Reads the hdf file as a numpy memmapped file, makes slicing a bit faster
    (From https://gist.github.com/rossant/7b4704e8caeb8f173084)

    Parameters
    ----------
    fn_hdf : str
        Path to preprocessed HDF5
    hdf_group : str
        Group to read from HDF5

    Returns
    -------
    data : array_like
        Data as a memory mapped array
    """
    # Define function for memmapping
    def _mmap_h5(path, h5path):
        with h5py.File(path, mode='r') as f:
            ds = f[h5path]
            # We get the dataset address in the HDF5 fiel.
            offset = ds.id.get_offset()
            # We ensure we have a non-compressed contiguous array.
            assert ds.chunks is None
            assert ds.compression is None
            assert offset > 0
            dtype = ds.dtype
            shape = ds.shape
        arr = np.memmap(path, mode='r', shape=shape,
                        offset=offset, dtype=dtype)
        return arr
    # Load data
    data = _mmap_h5(fn_hdf, hdf_group)

    return data
