#!/bin/bash
#SBATCH --job-name=ext
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=papri.beygi@gmail.com     # Where to send mail 
#SBATCH --cpus-per-task 4
#SBATCH --output /projects/ovcare/classification/parisa/slurm_scripts/logs/%j.out
#SBATCH --error /projects/ovcare/classification/parisa/slurm_scripts/logs/%j.out
#SBATCH -p gpu3090,rtx5000,gpu1080,gpu2080
#SBATCH --gres=gpu:4
#SBATCH --mem=80gb
#SBATCH --time=5-23:00:00 

cd /projects/ovcare/classification/parisa/projects/clam/CLAM

pwd
#nvidia-smi
#source activate clam
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home/pabeygi/anaconda3/envs/clam
conda info --envs

DIR_TO_COORDS=RESULTS_LUAD_test
DATA_DIRECTORY=/projects/ovcare/classification/parisa/datasets/TCGA/LUAD
CSV_FILE_NAME=RESULTS_LUAD_test/slide_filesnames.csv
FEATURES_DIRECTORY=FEATURES_DIRECTORY_LUAD_test

python extract_features_fp_debug.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs

#python extract_features_fp_debug.py --data_h5_dir RESULTS_BRCA_1024 --data_slide_dir /projects/ovcare/classification/parisa/datasets/TCGA_BRCA --csv_path RESULTS_BRCA_1024/slide_filesnames.csv --feat_dir FEATURES_DIRECTORY_BRCA_1024_Kimianet --batch_size 512 --slide_ext .svs --target_patch_size 512 --model_name kimianet --model_weight_path /projects/ovcare/classification/parisa/projects/pretrained/KimiaNetPyTorchWeights.pth

#python extract_instances_fp_debug.py --store_images --data_h5_dir RESULTS_BRCA_1024 --data_slide_dir /projects/ovcare/classification/parisa/datasets/TCGA_BRCA --csv_path RESULTS_BRCA_1024/slide_filesnames.csv --feat_dir IMAGE_DIRECTORY_BRCA_1024_ALL_500 --batch_size 512 --slide_ext .svs --target_patch_size 512 --num_patches 500
