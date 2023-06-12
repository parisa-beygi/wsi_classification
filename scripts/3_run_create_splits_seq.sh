#!/bin/bash
#SBATCH --job-name=split
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=papri.beygi@gmail.com     # Where to send mail 
#SBATCH --cpus-per-task 2
#SBATCH --output /projects/ovcare/classification/parisa/slurm_scripts/scripts/%j.out
#SBATCH --error /projects/ovcare/classification/parisa/slurm_scripts/scripts/%j.out
#SBATCH -p gpu3090,rtx5000,gpu1080,gpu2080
#SBATCH --gres=gpu:1
#SBATCH --mem=10gb
#SBATCH --time=23:00:00 

cd /projects/ovcare/classification/parisa/projects/clam/CLAM

pwd
#nvidia-smi
#source activate clam
source ~/anaconda3/etc/profile.d/conda.sh
#conda activate clam
conda activate /home/pabeygi/anaconda3/envs/clam
conda info --envs

TASK=task_1_tumor_vs_normal
SEED=1
DATASET_CSV_PATH=brca_dataset_csv/cibioportal_pan_brca_mut_vs_wt_PIK3CA.csv
LABLE_FRAC=1
VAL_FRAC=0.2
TEST_FRAC=0.2
K=5

python create_splits_seq.py --task $TASK --seed $SEED --label_frac $LABLE_FRAC --k $K --val_frac $VAL_FRAC --test_frac $TEST_FRAC --dataset_csv_path $DATASET_CSV_PATH

