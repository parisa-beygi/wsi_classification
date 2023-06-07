import sys
import json
import os
from glob import glob
import pandas as pd
import csv




if __name__ == "__main__":
    """
    This module takes 
    (1) sys.argv[1] the excel file containing labels
    (2) sys.argv[2] source directory of .svs files 
    (3) sys.argv[3]: gene
    """

    feature_dir = sys.argv[1]
    h5_dir = os.path.join(feature_dir, 'h5_files')

    h5_paths = glob(os.path.join(h5_dir, '*.h5'))
    # pt_paths = glob(os.path.join(pt_dir, '*.pt'))


    slide_ids = list(map(lambda x: x.split('/')[-1].split('.h5')[0], h5_paths))
    case_list = list(map(lambda x: x[:12], slide_ids))

    rows = [[case_list[i], slide_ids[i]] for i in range(len(case_list))]


    
    fields = ['case_id', 'slide_id']

    with open('brca_dataset_csv/brca_dataset_slide_case.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)



