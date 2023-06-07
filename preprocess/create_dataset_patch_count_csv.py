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
    (1) sys.argv[1] the dataset csv file containing columns: case_id,slide_id,label
        eg.:  /projects/ovcare/classification/parisa/projects/clam/CLAM/brca_dataset_csv/cibioportal_pan_brca_mut_vs_wt_TP53.csv
    (2) sys.argv[2] feature directory of dataset containing h5_files and pt_files
        eg.: /projects/ovcare/classification/parisa/projects/clam/CLAM/FEATURES_DIRECTORY_BRCA_use
    (3) sys.argv[3]: the dictory to save patch_count_csv
        eg.: /projects/ovcare/classification/parisa/projects/clam/CLAM/brca_dataset_csv/cibioportal_pan_brca_mut_vs_wt_TP53_patch.csv
    """
    dataset_csv_dir = sys.argv[1]
    feature_dir = sys.argv[2]
    out_csv_dir = sys.argv[3]

    # getting the file names, both .h5 and .pt
    h5_dir = os.path.join(feature_dir, 'h5_files')
    # pt_dir = os.path.join(feature_dir, 'pt_files')

    h5_paths = glob(os.path.join(h5_dir, '*.h5'))
    # pt_paths = glob(os.path.join(pt_dir, '*.pt'))

    h5_files = list(map(lambda x: x.split('/')[-1], h5_paths))
    # pt_files = list(map(lambda x: x.split('/')[-1], pt_paths))


    # read dataset csv file
    dataset_df = pd.read_csv(dataset_csv_dir)
    slide_ids = [s for s in dataset_df['slide_id']]

    patch_counts = []

    for s in slide_ids:
        h5_index = h5_files.index(s+'.h5')
        # pt_index = pt_files.index(s+'.pt')
        # assert h5_index == pt_index, f"h5 file index: {h5_index} != pt file index: {pt_index}"

        h5_path = h5_paths[h5_index]
        h5_file = h5py.File(h5_path, 'r')

        coords_num = h5_file['coords'].shape[0]
        features_num = h5_file['features'].shape[0]


        # pt_path = pt_paths[pt_index]
        # pt_tens = torch.load(pt_path)
        # features_tens_num = pt_tens.shape[0]

        assert coords_num == features_num, f"incompatible number of patches for slide {s}!"

        patch_counts.append(coords_num)
    

    final_df = pd.DataFrame({'slide_id': slide_ids, 'patch_count': patch_counts})
    final_df.to_csv(out_csv_dir)


