import sys
import json
import os
from glob import glob
import pandas as pd
import csv
import argparse



def get_case_slide_label_rows_cibioportal(annotation_file_dir, slide_dir, gene_list):
    annotations_df = pd.read_csv(annotation_file_dir, sep='\t')
    slide_paths = glob(os.path.join(slide_dir, '*'))
    slide_ids = list(map(lambda x: x.split('/')[-1].split('.svs')[0], slide_paths))

    rows = []
    for case_id in annotations_df['Patient ID'].values:
        l_list = []
        for gene in gene_list:
            label = annotations_df.loc[annotations_df['Patient ID'] == case_id][gene].values[0]
            l = 'WT' if label == 'no alteration' else 'MUT'
            l_list.append(l)

        for s in slide_ids:
            if case_id == s[:12]:
                slide_id = s
                rows.append([case_id, slide_id] + l_list)
    return rows

            


def get_case_slide_label_rows(annotation_file_dir, slide_dir, gene):
    annotations_df = pd.read_excel(annotation_file_dir, engine='openpyxl')

    slide_paths = glob(os.path.join(slide_dir, '*'))
    slide_ids = list(map(lambda x: x.split('/')[-1].split('.svs')[0], slide_paths))

    rows = []
    for case_id in annotations_df['PATIENT'].values:
        label = annotations_df.loc[annotations_df['PATIENT'] == case_id][f'{gene}_anymut'].values
        if len(label) == 0 or not isinstance(label[0], str):
            break
        
        label = label[0]
        for s in slide_ids:
            if case_id == s[:12]:
                slide_id = s
                rows.append([case_id, slide_id, label])
    return rows

if __name__ == "__main__":
    """
    This module takes 
    (1) sys.argv[1] the excel file containing labels
    (2) sys.argv[2] source directory of .svs files 
    (3) sys.argv[3]: gene
    """

    parser = argparse.ArgumentParser(description='Reading cbioportal gene mutation files into dataset csv file')
    parser.add_argument('--annotation_file_dir', type=str, default='brca_dataset_csv/brca_alterations_multi_4labels.tsv')
    parser.add_argument('--slide_dir', type=str, default='/projects/ovcare/classification/parisa/datasets/TCGA_BRCA', help = 'source directory of .svs files')
    parser.add_argument('--gene_list', nargs='+', default = ["TP53"], help='list of gene labels')
    parser.add_argument('--out_dir', type=str, default='brca_dataset_csv/brca_single_TP53.csv', help = 'out directory of result')

    args = parser.parse_args()


    # annotation_file_dir = sys.argv[1]
    # slide_dir = sys.argv[2]
    # gene = sys.argv[3]

    # rows = get_case_slide_label_rows(annotation_file_dir, slide_dir, gene)
    rows = get_case_slide_label_rows_cibioportal(args.annotation_file_dir, args.slide_dir, args.gene_list)

    
    fields = ['case_id', 'slide_id'] + args.gene_list
    with open(args.out_dir, 'w') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)



