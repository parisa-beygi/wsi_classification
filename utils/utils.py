import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import torch_optimizer
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
import torch_geometric

from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
import pandas as pd
from collections import Counter

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]


def collate_MIL_extended(batch):
	img = torch.stack([item[0] for item in batch], dim = 0)
	label = torch.LongTensor(np.array([item[1] for item in batch]))
	att = torch.stack([item[2] for item in batch])

	return [img, label, att]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 


def make_weights_for_multilabel_loss(dataset):
	N = float(len(dataset))
	count_classes = dataset.get_count_classes_labels()
	# weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	# weight_per_class = [N/count_classes[c] for c in range(len(count_classes))]                                                                                                     
	weight_per_label_per_class = N*(1 / count_classes)

	return torch.DoubleTensor(weight_per_label_per_class)


def get_split_loader_multilabel(split_dataset, training = False, testing = False, batch_size = 1, collate = 'ml'):
	"""
		return either the validation loader or training loader 
	"""
	# collate_fn = collate_MIL if collate == 'ml' else None
	if collate == 'ml':
		collate_fn = collate_MIL
	elif collate == 'ml_extended':
		collate_fn = collate_MIL_extended
	else:
		collate_fn = None

	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_fn, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_fn, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate_fn, **kwargs )

	return loader


def get_split_loader(split_dataset, training = False, testing = False, weighted = False, batch_size = 1, collate = 'ml'):
	"""
		return either the validation loader or training loader 
	"""
	# collate_fn = collate_MIL if collate == 'ml' else None
	if collate == 'ml':
		collate_fn = collate_MIL
	elif collate == 'ml_extended':
		collate_fn = collate_MIL_extended
	else:
		collate_fn = None

	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_fn, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_fn, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_fn, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate_fn, **kwargs )

	return loader


def get_split_loader_pyg(split_dataset, training = False, testing = False, weighted = False, batch_size = 1):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = torch_geometric.data.DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), **kwargs)	
			else:
				loader = torch_geometric.data.DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), **kwargs)
		else:
			loader = torch_geometric.data.DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = torch_geometric.data.DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), **kwargs )

	return loader


def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	elif args.opt == 'lookahead':
		base_opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
		optimizer = torch_optimizer.Lookahead(base_opt, k=5, alpha=0.5)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, pretrain_num, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_pretrain_ids = []
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class

			if pretrain_num is not None:
				pretrain_ids = possible_indices[:pretrain_num[c]] # pretrain ids
				possible_indices = np.setdiff1d(possible_indices, pretrain_ids) #indices of this class left after pretrain
				all_pretrain_ids.extend(pretrain_ids)

			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield all_pretrain_ids, sampled_train_ids, all_val_ids, all_test_ids


def print_label_combination(y_train, y_val, y_test, num_labels):
	print (pd.DataFrame({
		'train': Counter(str(combination) for row in get_combination_wise_output_matrix(y_train, order=num_labels) for combination in row),
		'val': Counter(str(combination) for row in get_combination_wise_output_matrix(y_val, order=num_labels) for combination in row),
		'test' : Counter(str(combination) for row in get_combination_wise_output_matrix(y_test, order=num_labels) for combination in row)
	}).T.fillna(0.0))


def generate_split_skmultilearn(y, samples, n_splits = 5,
	seed = 7):
	assert len(y) == samples, 'Incompatible number of samples and labels'
	np.random.seed(seed)
	y = pd.DataFrame(y)
	k_fold = IterativeStratification(n_splits=n_splits, order=2)
	X = np.zeros_like(y)
	fold = 0
	for train_val, test in k_fold.split(X, y):
		fold += 1
		y_train_val = y.loc[train_val]

		val_frac = 1/(n_splits - 1)
		X_train_val = np.zeros_like(y_train_val)

		k_fold_train_val = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold = [1- val_frac, val_frac])
		val_iloc, train_iloc = next(k_fold_train_val.split(X_train_val, y_train_val))
		
		train = y_train_val.iloc[train_iloc].index
		val = y_train_val.iloc[val_iloc].index

		print (f'Fold {fold}')
		print_label_combination(y.loc[train].to_numpy(), y.loc[val].to_numpy(), y.loc[test].to_numpy(), 2)
		print ()
		yield train, val, test		







def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))
	count_classes = dataset.get_count_classes()
	# weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight_per_class = [N/count_classes[c] for c in range(len(count_classes))]                                                                                                     

	weight = [0] * int(N)
	
	labels = dataset.getlabels()

	for idx in range(len(dataset)):   
		# y = dataset.getlabel(idx)                        
		y = labels[idx]
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)


def get_class_weights_for_multilabel_loss(dataset):
	N = float(len(dataset))
	count_label_classes = dataset.get_count_classes_labels()
	return torch.DoubleTensor(N/ count_label_classes) # 2x2 tensor each row corresponds to a label


def multilabel_CE(weights, logits, targets):
	loss = float(0)
	targets = torch.squeeze(targets)
	logits = torch.squeeze(logits)
	logits = torch.sigmoid(logits)

	for i in range(logits.shape[-1]):
		wn = weights[i][0]
		wp = weights[i][1]

		first_term = wp*targets[i]*torch.log(logits[i] + torch.finfo(torch.float32).eps)
		second_term = wn*(1 - targets[i])*torch.log(1 - logits[i] + torch.finfo(torch.float32).eps)

		loss -= (first_term + second_term)
	return loss





def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

