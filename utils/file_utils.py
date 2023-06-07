import pickle
import h5py

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    """
	e.g.: asset_dict = {'features': features, 'coords': coords}

    """
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def save_hdf5_groups(output_path, asset_dict, mode='a'):
    """
	e.g.: asset_dict = {'patch': {'prob': prob, 'label': label}}

    """
    for key in asset_dict:
        for sub_key in asset_dict[key]:
            new_key = key+'/'+sub_key
            save_hdf5(output_path, asset_dict={new_key: asset_dict[key][sub_key]}, mode = mode)
        
    return output_path

