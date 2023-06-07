from __future__ import print_function

import argparse
from genericpath import exists
import pdb
import os
import math

# internal imports
from utils.file_utils import save_hdf5_groups
from utils.utils import *
from utils.core_utils_patch import train
from datasets.patch_dataset_generic import Generic_Patch_Classification_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_err_acc = []
    all_val_err_acc = []



    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        val_results, test_results, val_dict, test_dict, split_counts = train(datasets, i, args)

        #write results
        dirname = os.path.join(args.results_dir, 'split_{}_results'.format(i))
        os.makedirs(dirname, exist_ok=True)
        val_results.to_csv(os.path.join(dirname, 'summary_val.csv'))
        test_results.to_csv(os.path.join(dirname, 'summary_test.csv'))
        split_counts.to_csv(os.path.join(dirname, 'split_counts.csv'))

        save_hdf5_groups(os.path.join(dirname, 'val_prob_label.h5'), val_dict)
        save_hdf5_groups(os.path.join(dirname, 'test_prob_label.h5'), test_dict)



# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for Patch Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--model_type', type=str, choices=['baseline', 'baseline_finetune'], default='baseline_finetune', 
                    help='type of model (default: kather)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'medium', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str)
### Patch-based learning specific options
parser.add_argument('--num_patches', type=int, default=500, help='the number of sample patches per slide')
parser.add_argument('--batch_size', type=int, default=1, help='the batch size used for loading data')

parser.add_argument('--dataset_csv_path', type=str, default='brca_dataset_csv/cibioportal_pan_brca_mut_vs_wt_TP53.csv', 
                    help='the path to dataset vsc file containing at least 3 columns: case_id, slide_id, slide-level labels')

#patch_csv_path = 'brca_dataset_csv/cibioportal_pan_brca_mut_vs_wt_TP53_patch_npatches.csv' if args.model_type == 'baseline_finetune' else 'brca_dataset_csv/cibioportal_pan_brca_mut_vs_wt_TP53_patch_all.csv'
parser.add_argument('--dataset_patches_csv_path', type=str, default='brca_dataset_csv/cibioportal_pan_brca_mut_vs_wt_TP53_patch_npatches.csv', 
                    help='the path to dataset vsc file containing at least 3 columns: case_id, slide_id, slide-level labels')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'batch_size': args.batch_size,
            'num_patches': args.num_patches}



print('\nLoad Dataset')

if 'mut_vs_wt' in args.task:
    args.n_classes=2
    dataset = Generic_Patch_Classification_Dataset(csv_path = args.dataset_csv_path,
                            patch_csv_path= args.dataset_patches_csv_path,
                            data_dir= os.path.join(args.data_root_dir),
                            num_patches = args.num_patches,
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'WT':0, 'MUT':1},
                            patient_strat=False,
                            ignore=[])
else:
    raise NotImplementedError                            

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir, exist_ok=True)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
