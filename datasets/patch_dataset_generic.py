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

from utils.utils import generate_split, nth


class Generic_Patch_Classification_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv',
        patch_csv_path = 'patch_dataset_csv/wsi_patch_count.csv',
        data_dir = 'features',
        num_patches = 500,
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
            patch_csv_path (string): Path to the csv file with slide patch count
            patch_num (int): Number of patches selected from each slide
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """

        self.data_dir = data_dir
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.num_patches = num_patches
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        if not label_col:
            label_col = 'label'
        self.label_col = label_col
        
        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

        self.slide_data = slide_data

        np.random.seed(seed)
        self.patch_data = self.create_patch_data(patch_csv_path)

        ###shuffle data
        if shuffle:
            arr = self.patch_data.values
            np.random.shuffle(arr)
            self.patch_data = pd.DataFrame(arr, columns=['slide_id', 'case_id', 'patch_index', 'label'])

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()


    def create_patch_data(self, csv_path):
        patch_data = []
        num_patches_df = pd.read_csv(csv_path)
        for slide_id, case_id, label in np.array(self.slide_data[['slide_id', 'case_id', 'label']]):
            loc = num_patches_df[num_patches_df['slide_id'] == slide_id].index.tolist()
            assert len(loc) == 1, f"slide id {slide_id} has {len(loc)} etries in {csv_path}!"
            num = num_patches_df['patch_count'][loc].values[0]
            if num < self.num_patches:
                continue
            patch_indices = np.random.choice(num, self.num_patches, replace=False)
            for i in patch_indices:
                patch_data.append((slide_id, case_id, i, label))
        return pd.DataFrame(patch_data, columns=['slide_id', 'case_id', 'patch_index', 'label'])


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

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, len(np.where(self.patient_data['label'] == i)[0])))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, len(np.where(self.slide_data['label'] == i)[0])))
            print('Patch-LVL; Number of samples registered in class %d: %d' % (i, self.num_patches*len(np.where(self.slide_data['label'] == i)[0])))


    def __len__(self):
        return len(self.patch_data)

    def get_split_from_df(self, all_splits, split_key='train'):
        if split_key not in all_splits.columns:
            return None

        if split_key == 'train' and 'pretrain' in all_splits.columns:
            split = all_splits[['pretrain', 'train']]
            split = split['train'].append(split['pretrain']).reset_index(drop=True)
        else:
            split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            patch_data_slice = self.patch_data[self.patch_data['slide_id'].isin(df_slice['slide_id'])]
            patch_data_slice.reset_index()

            split = Generic_Split(df_slice, patch_data_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        else:
            split = None
		
        return split



    def return_splits(self, csv_path=None):
        assert csv_path
        all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
        pretrain_split = self.get_split_from_df(all_splits, 'pretrain')
        train_split = self.get_split_from_df(all_splits, 'train')
        val_split = self.get_split_from_df(all_splits, 'val')
        test_split = self.get_split_from_df(all_splits, 'test')
			
        return pretrain_split, train_split, val_split, test_split


    def __getitem__(self, index):
        slide_id, _, patch_index, label = self.patch_data.iloc[index]
        # loc = self.slide_data[self.slide_data['slide_id'] == slide_id].index.tolist()
        # assert len(loc) == 1, f"slide id {slide_id} has {len(loc)} etries!"
        # label = self.slide_data['label'][loc].values[0]
        
        slide_full_path = os.path.join(self.data_dir, 'h5_files', '{}.h5'.format(slide_id))
        with h5py.File(slide_full_path, 'r') as h5_file:
            patch = h5_file['features'][patch_index, ]
            patch = torch.from_numpy(patch)
 
        return patch, label

    def getlabel(self, ids):
        _, _, label = self.patch_data.iloc[ids]
        # loc = self.slide_data[self.slide_data['slide_id'] == slide_id].index.tolist()
        # assert len(loc) == 1, f"slide id {slide_id} has {len(loc)} etries!"
        # label = self.slide_data['label'][loc].values[0]

        return label

    def getlabels(self):
        return self.patch_data['label'].tolist()

    def get_count_classes(self):
        return [self.num_patches*len(self.slide_cls_ids[i]) for i in range(self.num_classes)]

class Generic_Split(Generic_Patch_Classification_Dataset):
    def __init__(self, slide_data, patch_data, data_dir=None, num_classes=2):
        self.slide_data = slide_data
        self.patch_data = patch_data
        self.data_dir = data_dir
        self.num_classes = num_classes

        self.num_patches = len(self.patch_data) // len(self.slide_data)

        self.patient_data_prep(patient_voting='max')
        self.cls_ids_prep()

    def __len__(self):
        return len(self.patch_data)
