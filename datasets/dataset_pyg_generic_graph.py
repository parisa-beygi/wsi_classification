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

# from torch.utils.data import Dataset
import h5py

from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans

from utils.utils import generate_split, nth

from sklearn.neighbors import NearestNeighbors
import torch_geometric
from torch_geometric.data import Data

from torch_geometric.data import Dataset
import time

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
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
		# self.data_dir = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

		super().__init__(root, transform, pre_transform, pre_filter)


	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]		
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

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

	def len(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))
		print ('Class weights are: ', self.get_class_weights())
	
	def get_class_weights(self):
		total = 0
		nums = []
		for i in range(self.num_classes):
			num_class_samples = self.slide_cls_ids[i].shape[0]
			total += num_class_samples
			nums.append(num_class_samples)
		return list(map(lambda x: 1 - x/total, nums))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

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
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
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
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
			
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
		return self.slide_data['label'][ids]

	def getlabels(self):
		return self.slide_data['label'].tolist()

	def get(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

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

	def get_count_classes(self):
		return [len(self.slide_cls_ids[i]) for i in range(self.num_classes)]

class Generic_Graph_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self, data_dir, num_clusters, k_farthes_in_cluster, **kwargs):	
		self.data_dir = data_dir
		self.use_h5 = True
		self.num_clusters = num_clusters
		self.k_farthes_in_cluster = k_farthes_in_cluster
		super(Generic_Graph_Dataset, self).__init__(**kwargs)

	def load_from_h5(self, toggle):
		self.use_h5 = toggle


	def create_wheel_edges(self, centroid_index, wheel_num_points, total_num_centroids):
		begin = total_num_centroids + centroid_index*wheel_num_points
		end = begin + wheel_num_points

		# print (begin, end)

		l = []
		for i in range(begin, end):
			l.append((i, centroid_index))
			l.append((centroid_index, i))

			l.append((i, i))
			if i != end - 1:
				l.append((i, i+1))
				l.append((i+1, i))
		
		l.append((begin, end - 1))
		l.append((end - 1, begin))

		# print (len(l))
		# print (l)
		return l

	def construct_graph(self, data):
		# data : n, d
		# perform clustering
		kmeans = KMeans(n_clusters=self.num_clusters, verbose=1)
		labels = kmeans.fit_predict(data)
		cents = kmeans.cluster_centers_

		edges = []

		centroids = []
		perpheries = []
		# create the wheel graphs
		for i, c in enumerate(cents):
			in_cluster_indices = np.nonzero(labels == i)[0]
			in_cluster = data[in_cluster_indices]
			dist = distance_matrix(in_cluster, c[np.newaxis, :])

			furthest_in_cluster_new_indices = np.argsort(dist[:, 0])[::-1][:self.k_farthes_in_cluster]

			furthest_in_cluster_indices = in_cluster_indices[furthest_in_cluster_new_indices]
			furthest_in_cluster = data[furthest_in_cluster_indices]

			centroids.append(c)
			perpheries.append(furthest_in_cluster)		

		centroids = np.array(centroids)
		perpheries = np.array(perpheries)

		cnodes = []
		for i, c in enumerate(centroids):
			cnodes.append(c)

		cedges = [(i, j) for i in range(len(centroids)) for j in range(len(centroids))]

		pnodes = []
		pedges = []

		num_centroids = len(centroids)
		for i in range(len(centroids)):
			pnodes.append(perpheries[i])

			num = len(perpheries[i])
			wheel_edges = self.create_wheel_edges(i, num, num_centroids)

			pedges.extend(wheel_edges)


		cnodes = np.array(cnodes)
		cedges = np.array(cedges)

		pnodes = np.concatenate(pnodes, axis=0)
		pedges = np.array(pedges)

		allnodes = torch.from_numpy(np.concatenate((cnodes, pnodes), axis=0))
		alledges = torch.from_numpy(np.concatenate((cedges, pedges), axis=0))

		alledges = torch.transpose(alledges, 0, 1)

		return allnodes, alledges

	
	@property
	def raw_file_names(self):
		return 'cluster_graph_dataset'

	@property
	def processed_file_names(self):
		return [f'data_{i}.pt' for i in range(len(self.slide_data))]

	def download(self):
		pass

		
	def process(self):
		data_size = len(self.slide_data)
		for idx in range(data_size):
			slide_id = self.slide_data['slide_id'][idx]
			label = self.slide_data['label'][idx]
			full_path = os.path.join(self.data_dir,'h5_files','{}.h5'.format(slide_id))

			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			# features = torch.from_numpy(features)
			start_time = time.time()
			all_nodes, all_edges = self.construct_graph(features)
			data = Data(x=all_nodes, edge_index=all_edges, y = label)
			elapsed = time.time() - start_time
			print (f'Constructed graph in {elapsed} sec.')

			torch.save(data, os.path.join(self.processed_dir, f'data_{slide_id}.pt'))


	def get(self, idx):
		data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
		return data




class Generic_Split(Generic_Graph_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2):
		self.use_h5 = True
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes


		self.patient_data_prep(patient_voting='max')
		self.cls_ids_prep()

		# self.slide_cls_ids = [[] for i in range(self.num_classes)]
		# for i in range(self.num_classes):
		# 	self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def len(self):
		return len(self.slide_data)
		

