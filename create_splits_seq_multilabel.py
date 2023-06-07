import pdb
import os
import pandas as pd
from datasets.dataset_generic_multilabel import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--dataset_csv_path', type=str, default='brca_dataset_csv/cibioportal_pan_brca_mut_vs_wt_TP53.csv', 
                    help='the path to dataset vsc file containing at least 3 columns: case_id, slide_id, slide-level labels')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str)
parser.add_argument('--labels', nargs='+', default=["TP53"], help='labels for multi-label classification')


args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
if 'mut_vs_wt' in args.task:
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.dataset_csv_path,
                            label_col = args.labels,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'WT':0, 'MUT':1},
                            patient_strat=True,
                            ignore=[])                            

else:
    raise NotImplementedError



if __name__ == '__main__':    
    data_label_id = args.dataset_csv_path.split('/')[-1].split('.')[0]
    columns = ['train', 'val', 'test']
    split_dir = 'splits/'+ str(args.task) + f'_{data_label_id}' + '_k{}'.format(args.k) + '_s{}'.format(args.seed)
    os.makedirs(split_dir, exist_ok=True)
    dataset.create_splits(k = args.k)
    for i in range(args.k):
        dataset.set_splits()
        descriptor_df = dataset.test_split_gen(return_descriptor=True)
        splits = dataset.return_splits(from_id=True)
        save_splits(splits, columns, os.path.join(split_dir, 'splits_{}.csv'.format(i)))
        save_splits(splits, columns, os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
        descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



