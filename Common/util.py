from __future__ import absolute_import
from keras.models import load_model, save_model
import tempfile
import os
import h5py

class Util:
    """
    Utility module
    """
    @classmethod
    def save_model_to_hdf5_group(cls, model, f):
        # Use Keras save_model to save the full model (including optimizer
        # state) to a file.
        # Then we can embed the contents of that HDF5 file inside ours.
        tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
        try:
            os.close(tempfd)
            print(tempfname)
            save_model(model, tempfname + '.h5' , overwrite=True)
            serialized_model = h5py.File(tempfname + '.h5', 'r')
            root_item = serialized_model.get('/')
            serialized_model.copy(root_item, f, 'kerasmodel')
            serialized_model.close()
        finally:
            os.unlink(tempfname)

    @classmethod
    def load_model_from_hdf5_group(cls, f, custom_objects=None):
        # Extract the model into a temporary file. Then we can use Keras
        # load_model to read it.
        tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
        tempfname += '.h5'
        try:
            os.close(tempfd)
            serialized_model = h5py.File(tempfname, 'w')
            root_item = f.get('kerasmodel')
            for attr_name, attr_value in root_item.attrs.items():
                serialized_model.attrs[attr_name] = attr_value
            for k in root_item.keys():
                f.copy(root_item.get(k), serialized_model, k)
            serialized_model.close()
            return load_model(tempfname, custom_objects=custom_objects)
        finally:
            os.unlink(tempfname)