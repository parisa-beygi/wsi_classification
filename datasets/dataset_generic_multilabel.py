from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split_skmultilearn, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	split_datasets = [split for split in split_datasets if split is not None]
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = column_keys)

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		assert list(slide_data.columns)[2:] == label_col, f"Incompatible list of labels given : {list(slide_data.columns)[2:]} - {label_col}"
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		self.label_prep()

		if print_info:
			self.summarize()


	def label_prep(self):
		self.patient_labels = self.patient_data[self.label_col].to_numpy()
		self.slide_labels = self.slide_data[self.label_col].to_numpy()



	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		patient_res = []
		for j in range(len(self.label_col)):
			patient_res_classes = []
			for i in range(self.num_classes):
				patient_res_classes.append(np.where(self.patient_data[self.label_col[j]] == i)[0])
			patient_res.append(patient_res_classes)
		self.patient_cls_ids = patient_res

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		slide_res = []
		for j in range(len(self.label_col)):
			slide_res_classes = []
			for i in range(self.num_classes):
				slide_res_classes.append(np.where(self.slide_data[self.label_col[j]] == i)[0])
			slide_res.append(slide_res_classes)
		self.slide_cls_ids = slide_res

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0

			each_patient_labels = []
			for label in self.label_col:
				label = self.slide_data[label][locations].values
				if patient_voting == 'max':
					label = label.max() # get patient label (MIL convention)
				elif patient_voting == 'maj':
					label = stats.mode(label)[0]
				else:
					raise NotImplementedError
				each_patient_labels.append(label)

			patient_labels.append(each_patient_labels)
		
		patient_labels = np.array(patient_labels)
		data_dict = {'case_id':patients}
		for j in range(len(self.label_col)):
			data_dict.update({self.label_col[j]: patient_labels[:, j]})

		
		self.patient_data = pd.DataFrame(data_dict)

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		mask = data[label_col].isin(ignore)
		final_mask = True
		for c in label_col:
			final_mask = final_mask & mask[c]

		data = data[~final_mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			for label in label_col:
				key = data.loc[i, label]
				data.at[i, label] = label_dict[key]

		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		for j in range(len(self.label_col)):
			print(f'{self.label_col[j]} slide-level counts: ', '\n', self.slide_data[self.label_col[j]].value_counts(sort = False))		
			for i in range(self.num_classes):
				print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[j][i].shape[0]))
				print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[j][i].shape[0]))

	def create_splits(self, k = 3):
		settings = {
					'n_splits' : k, 
					'seed': self.seed,
					}

		if self.patient_strat:
			settings.update({'y' : self.patient_labels, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'y' : self.slide_labels, 'samples': len(self.slide_data)})

		self.split_gen = generate_split_skmultilearn(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): 
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		if split_key not in all_splits.columns:
			return None
			
		split = all_splits[split_key]
		
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, label_col=self.label_col)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):
		if from_id:

			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes, label_col=self.label_col)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes, label_col=self.label_col)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes, label_col=self.label_col)
			
			else:
				test_split = None
			
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data[self.label_col].loc[ids]

	def getlabels(self):
		return self.slide_data[self.label_col].tolist()

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [f'{self.label_col[j]}_{list(self.label_dict.keys())[list(self.label_dict.values()).index(i)]}' for j in range(len(self.label_col)) for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)


		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		all_labels = self.getlabel(self.train_ids)
		counter = 0
		for c in self.label_col:
			labels = all_labels[c]
			unique, counts = np.unique(labels, return_counts=True)
			for u in range(len(unique)):
				print('{} - number of samples in cls {}: {}'.format(c, unique[u], counts[u]))
				if return_descriptor:
					df.loc[index[counter], 'train'] = counts[u]
				counter +=1
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		all_labels = self.getlabel(self.val_ids)
		counter = 0
		for c in self.label_col:
			labels = all_labels[c]
			unique, counts = np.unique(labels, return_counts=True)
			for u in range(len(unique)):
				print('{} - number of samples in cls {}: {}'.format(c, unique[u], counts[u]))
				if return_descriptor:
					df.loc[index[counter], 'val'] = counts[u]
				counter +=1

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		all_labels = self.getlabel(self.test_ids)
		counter = 0
		for c in self.label_col:
			labels = all_labels[c]
			unique, counts = np.unique(labels, return_counts=True)
			for u in range(len(unique)):
				print('{} - number of samples in cls {}: {}'.format(c, unique[u], counts[u]))
				if return_descriptor:
					df.loc[index[counter], 'test'] = counts[u]
				counter +=1

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)

	def get_count_classes_labels(self):
		res = []
		for j in range(len(self.label_col)):
			res_label = []
			for i in range(self.num_classes):
				res_label.append(len(self.slide_cls_ids[j][i]))
			res.append(res_label)
		return np.array(res)

class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data[self.label_col].loc[idx].to_numpy()
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
				features = torch.load(full_path)
				return features, label
			
			else:
				return slide_id, label

		else:
			full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, coords


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, label_col, data_dir=None, num_classes=2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.label_col = label_col

		self.patient_data_prep(patient_voting='max')
		self.cls_ids_prep()

		# self.slide_cls_ids = [[] for i in range(self.num_classes)]
		# for i in range(self.num_classes):
		# 	self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
		


