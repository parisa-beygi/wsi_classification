import sys
import json
import os
from glob import glob
import pandas as pd
import csv
import h5py
import torch

if __name__ == "__main__":
    """
    This module takes 
    (1) sys.argv[1] feature directory of dataset containing h5_files and pt_files
        eg.: /projects/ovcare/classification/parisa/projects/clam/CLAM/FEATURES_DIRECTORY_BRCA_use
    (2) sys.argv[2]: the dictory to save patch_count_csv
        eg.: /projects/ovcare/classification/parisa/projects/clam/CLAM/brca_dataset_csv/cibioportal_pan_brca_mut_vs_wt_TP53_patch.csv
    """
    feature_dir = sys.argv[1]
    out_csv_dir = sys.argv[2]

    # getting the file names, both .h5 and .pt
    h5_dir = os.path.join(feature_dir, 'h5_files')
    # pt_dir = os.path.join(feature_dir, 'pt_files')

    h5_paths = glob(os.path.join(h5_dir, '*.h5'))
    # pt_paths = glob(os.path.join(pt_dir, '*.pt'))


    slide_ids = list(map(lambda x: x.split('/')[-1].split('.h5')[0], h5_paths))

    patch_counts = []

    for h5_path in h5_paths:
        with h5py.File(h5_path, 'r') as h5_file:
            features_num = h5_file['features'].shape[0]


        patch_counts.append(features_num)
    

    final_df = pd.DataFrame({'slide_id': slide_ids, 'patch_count': patch_counts})
    final_df.to_csv(out_csv_dir)


