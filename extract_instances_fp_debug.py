import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=False, apply_transform = False, 
	custom_downsample=1, target_patch_size=-1, store_images = False, num_patches = 0):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, apply_transform = apply_transform,
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))


	if store_images and num_patches:
		n_bins = len(loader)

		if n_bins > 1:
			num_patches_batch = num_patches // (n_bins - 1)
			bin_counts = n_bins*[num_patches_batch]

			total_num_patches = len(dataset)
			last_batch_size = total_num_patches - (n_bins - 1)*batch_size
			remain = num_patches - num_patches_batch*(n_bins - 1)

			if remain < last_batch_size and remain <= num_patches_batch:
				bin_counts[-1] = remain
			elif remain < last_batch_size:
				remain -= num_patches_batch
				for i in range(remain):
					bin_counts[i] += 1
			else:
				bin_counts[-1] = last_batch_size
				remain -= last_batch_size
				k = remain // (n_bins - 2)
				r = remain % (n_bins - 2)
				for i in range(n_bins - 2):
					bin_counts[i] += k
				bin_counts[-2] += r

		else:
			bin_counts = [min(len(dataset), num_patches)]



	mode = 'w'
	time_start = time.time()
	model_time = 0
	save_time = 0
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				time_elapsed = time.time() - time_start
				print('batch {}/{}, {} files processed in {} s.'.format(count, len(loader), count * batch_size, time_elapsed))
				# print (f'model eval time is {model_time} s.')
				# print (f'save time is {save_time} s.')
				

			if store_images:
				batch = batch.numpy()
				if num_patches:
					indices = np.random.choice(batch.shape[0], bin_counts[count], replace=False)
					if not indices.size:
						continue
					batch = batch[indices]
					coords = coords[indices]
				asset_dict = {'features': batch, 'coords': coords}
				save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
				mode = 'a'
			else:
				model_time = 0
				save_time = 0
				time_start = time.time()
				batch = batch.to(device, non_blocking=True)
				
				model_start_time = time.time()
				features = model(batch)
				model_elapsed_time = time.time() - model_start_time
				model_time += model_elapsed_time
				features = features.cpu().numpy()

				asset_dict = {'features': features, 'coords': coords}
				save_start_time = time.time()
				save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
				save_elapsed_time = time.time() - save_start_time
				save_time += save_elapsed_time
				mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)

### specific options for storing num_patches patch images per slide
parser.add_argument('--store_images', action='store_true', default=False, help='Enable storing patch images instead of patch features')
parser.add_argument('--num_patches', type=int, default=0, help='extract all patches')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')


args = parser.parse_args()

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
	dest_files_h5 = os.listdir(os.path.join(args.feat_dir, 'h5_files'))


	print('loading model checkpoint')
	model = resnet50_baseline(pretrained=True)
	model = model.to(device)
	
	# print_network(model)
	print (f'number of devices: {torch.cuda.device_count()}')
	print (f'device: {device}')
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and (slide_id+'.pt' in dest_files or slide_id+'.h5' in dest_files_h5):
			print('skipped {}'.format(slide_id))
			continue

		from glob import glob
		if h5_file_path not in glob(os.path.join(args.data_h5_dir, 'patches', '*.h5')):
			print (f'No h5 file found, skipped {slide_id}')
			continue
		
		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)

		# if h5_file_path in glob(os.path.join(args.feat_dir, 'h5_files')):
		# 	print (f'Already created {output_path}')
		# 	continue

		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
		store_images = args.store_images, num_patches = args.num_patches)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		# features = file['features'][:]
		# print('features size: ', features.shape)
		# print('coordinates size: ', file['coords'].shape)
		# features = torch.from_numpy(features)
		# bag_base, _ = os.path.splitext(bag_name)
		# torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))


