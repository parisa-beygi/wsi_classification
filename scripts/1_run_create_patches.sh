#!/bin/bash
#SBATCH --job-name=create_p
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=papri.beygi@gmail.com     # Where to send mail 
#SBATCH --cpus-per-task 4
#SBATCH --output /projects/ovcare/classification/parisa/slurm_scripts/scripts/%j.out
#SBATCH --error /projects/ovcare/classification/parisa/slurm_scripts/scripts/%j.out
#SBATCH -p gpu3090,rtx5000,gpu1080,gpu2080
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH --time=5-23:00:00 

cd /projects/ovcare/classification/parisa/projects/clam/CLAM

pwd
#nvidia-smi
#source activate clam
source ~/anaconda3/etc/profile.d/conda.sh
#conda activate clam
conda activate /home/pabeygi/anaconda3/envs/clam
conda info --envs

DATA_DIRECTORY=/projects/ovcare/classification/parisa/datasets/TCGA/LUAD
RESULTS_DIRECTORY=RESULTS_LUAD

python create_patches_fp.py --source $DATA_DIRECTORY --save_dir $RESULTS_DIRECTORY --patch_size 256 --preset tcga.csv --seg --patch --stitch

#python extract_features_fp_debug.py --data_h5_dir RESULTS_resolved --data_slide_dir /projects/ovcare/classification/parisa/datasets/TCGA-COAD --csv_path RESULTS_resolved/slide_filesnames.csv --feat_dir FEATURES_DIRECTORY_use --batch_size 512 --slide_ext .svs
