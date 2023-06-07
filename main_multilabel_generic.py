from __future__ import print_function

import argparse
from genericpath import exists
import pdb
import os
import math

# internal imports
from utils.file_utils import save_hdf5_groups
from utils.utils import *
from utils.core_utils_multilabel_generic import train
from datasets.dataset_generic_multilabel import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from datasets.dataset_generic_knn_graph_multilabel import Generic_KNN_Graph_Dataset

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
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        val_results, test_results, val_dict, test_dict, split_counts_list = train(datasets, i, args)

        #write results
        dirname = os.path.join(args.results_dir, 'split_{}_results'.format(i))
        os.makedirs(dirname, exist_ok=True)
        val_results.to_csv(os.path.join(dirname, 'summary_val.csv'))
        test_results.to_csv(os.path.join(dirname, 'summary_test.csv'))
        for si, split_counts in enumerate(split_counts_list):
            split_counts.to_csv(os.path.join(dirname, f'split_counts_{si}.csv'))


        save_hdf5_groups(os.path.join(dirname, 'val_prob_label.h5'), val_dict)
        save_hdf5_groups(os.path.join(dirname, 'test_prob_label.h5'), test_dict)


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
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
                    +'instead of infering from the task_kfold_seed argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')

parser.add_argument('--att_branch', type=str, choices=['single', 'multi'], default='single', 
                    help='number of attention branches  (default: single)')
parser.add_argument('--att_model_type', type=str, choices=['clam', 'clam_gated', 'ABMIL', 'ABMIL_gated', 'VarMIL', 'LocalViT'], default='clam', 
                    help='type of attention model (default: clam)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')

parser.add_argument('--task', type=str)
parser.add_argument('--labels', nargs='+', default=["TP53"], help='labels for multi-label classification')
parser.add_argument('--batch_size', type=int, default=1, help='the batch size used for loading data')
parser.add_argument('--dataset_csv_path', type=str, default='brca_dataset_csv/cibioportal_pan_brca_mut_vs_wt_TP53.csv', 
                    help='the path to dataset vsc file containing at least 3 columns: case_id, slide_id, slide-level labels')
### Attention model's specific options
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--k_neighbors', type=int, default=64, help='numbr of knn neighbors for local graphs')

### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')





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
            'seed': args.seed,
            'att_model_type': args.att_model_type,
            'att_branch': args.att_branch,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'opt': args.opt,
            'labels': args.labels}

settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

    assert args.subtyping 
        
elif 'mut_vs_wt' in args.task:
    args.n_classes=2
    args.n_labels=len(args.labels)
    if args.att_model_type == 'LocalViT':
        dataset = Generic_KNN_Graph_Dataset(csv_path = args.dataset_csv_path,
                                k_neighbors=args.k_neighbors,
                                label_col = args.labels,
                                data_dir= os.path.join(args.data_root_dir),
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'WT':0, 'MUT':1},
                                patient_strat=False,
                                ignore=[])
    else:
        dataset = Generic_MIL_Dataset(csv_path = args.dataset_csv_path,
                                label_col = args.labels,
                                data_dir= os.path.join(args.data_root_dir),
                                shuffle = False, 
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
    args.split_dir = os.path.join('splits', args.task+str(args.task) + '_k{}'.format(args.k) + '_s{}'.format(args.seed))
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


