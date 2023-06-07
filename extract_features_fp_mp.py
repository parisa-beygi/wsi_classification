import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time

from tqdm import tqdm
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from utils.added_utils import select_slides, get_slide_names
from PIL import Image
import h5py
import openslide

# multiprocessing
import torch.multiprocessing as mp
import psutil

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def compute_w_loader(file_path, output_path, wsi_path, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, send_end=None):
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
	output_stream = []

	wsi = openslide.open_slide(wsi_path)
	
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))
		output_stream.append('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	time_start = time.time()
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():
			if count % print_every == 0:
				time_elapsed = time.time() - time_start
				print('batch {}/{}, {} files processed in {} s'.format(count, len(loader), count * batch_size, time_elapsed))
				output_stream.append('batch {}/{}, {} files processed in {} s'.format(count, len(loader), count * batch_size, time_elapsed))
				time_start = time.time()
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return send_end.send(output_path, "\n".join(output_stream))



def produce_args(cur_slide_names, args):
	res_args = []
	for slide_id in cur_slide_names:
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)

		# if not args.no_auto_skip and slide_id+'.pt' in dest_files:
		# 	print('skipped {}'.format(slide_id))
		# 	continue

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		# wsi = openslide.open_slide(slide_file_path)

		slide_args = (h5_file_path, output_path, slide_file_path)
		res_args.append(slide_args)
	return res_args



parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--slide_idx', type=int, default=1)
parser.add_argument('--n_process', type=int, default=psutil.cpu_count())

args = parser.parse_args()




if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	# bags_dataset = Dataset_All_Bags(csv_path)
	slide_names = get_slide_names(csv_path)
	n_process = args.n_process

	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	model = resnet50_baseline(pretrained=True)
	model = model.to(device)
	
	# print_network(model)
	if torch.cuda.device_count() > 1:
		print (f'Number of GPUs available are {torch.cuda.device_count()}')
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(slide_names)

	if args.slide_idx != -1:
		slide_names = select_slides(slide_names, args.slide_idx, n_process)
	mp.set_start_method('spawn')


	prefix = f'Processing {total} slides: '
	results_to_write = []
	for idx in tqdm(range(0, total, n_process), desc=prefix, dynamic_ncols=True):
		cur_slide_names = slide_names[idx:idx + n_process]
		slides_args = produce_args(cur_slide_names, args)

		processes = []
		recv_end_list = []
		for slide_args in slides_args:
			recv_end, send_end = mp.Pipe(False)
			recv_end_list.append(recv_end)

			# p = mp.Process(target=compute_w_loader,
			# 	args=(slide_args[0], slide_args[1], slide_args[2], slide_args[3], slide_args[4], slide_args[5], slide_args[6], slide_args[7], slide_args[8], slide_args[9], send_end))

			p = mp.Process(target=compute_w_loader, args=(slide_args[0], slide_args[1], slide_args[2], model, args.batch_size, 1, 20, True, args.custom_downsample, args.target_patch_size, send_end))

			print (f'Starting process for {slide_args[0]}')
			p.start()
			processes.append(p)
		for p in processes:
			p.join()

		results_to_write.extend(map(lambda x: x.recv(), recv_end_list))


	for slide_id, res in zip(slide_names, results_to_write):
		output_file_path, output_log = res[0], res[1]

		with open(os.path.join(args.feat_dir, 'log.txt'), 'w') as f:
			f.write(output_log)

		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_name = slide_id+'.h5'
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
