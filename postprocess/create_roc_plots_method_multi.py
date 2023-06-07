import matplotlib.pylab as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse

import h5py
from glob import glob
import os
from matplotlib.pyplot import cm

parser = argparse.ArgumentParser(description='Configurations for plotting ROC')
# parser.add_argument('--gene', type=str, choices = ['tp53', 'PIK3CA'], default='tp53')
parser.add_argument('--gene', nargs='+', default=['tp53', 'PIK3CA', 'MAP3K1', 'MAP2K4'], help='labels for multi-label classification')

parser.add_argument('--experiment_root_dir', type=str, default="/projects/ovcare/classification/parisa/projects/clam/CLAM/results_new/mut_brca_{}_k5_MULTI_4labels", 
                    help='data directory')


parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--s', type=int, default=3, help='number of seeds (default: 3)')
parser.add_argument('--result_dir', type=str, default="/projects/ovcare/classification/parisa/projects/clam/CLAM/results_new_plots/ABMIL_gated_sb", 
                    help='result directory to store ROC plots')

# parser.add_argument('--methods', type=int, nargs='+', default =["VOS-ABMIL", "CLUST-ABMIL", "VOSsb-ABMIL", "CLUSTsb-ABMIL", "ABMIL-attention"] , help='Method names to plot ROC for')                    
parser.add_argument('--methods', type=int, nargs='+', default =["VOS-ABMIL_gated", "CLUST-ABMIL_gated", "VOSsb-ABMIL_gated", "CLUSTsb-ABMIL_gated", "ABMIL-gated_attention"] , help='Method names to plot ROC for')                    
# parser.add_argument('--methods', type=int, nargs='+', default =["VOS-clam", "CLUST-clam", "VOSsb-clam", "CLUSTsb-clam", "CLAM-attention-noinst"] , help='Method names to plot ROC for')                    
# parser.add_argument('--methods', type=int, nargs='+', default =["VOS-clam_gated", "CLUST-clam_gated", "VOSsb-clam_gated", "CLUSTsb-clam_gated", "CLAM-attention-gated-noinst"] , help='Method names to plot ROC for')                    


args = parser.parse_args()



if not os.path.exists(args.result_dir):
   # Create a new directory because it does not exist
   os.makedirs(args.result_dir)
   print("The new directory is created!")


def plot_roc_kfolds_nseeds(gene_index, gene):
    color = iter(cm.rainbow(np.linspace(0, 1, len(args.methods))))


    for method in args.methods:

        experiment_root_dir = args.experiment_root_dir
        experiment_root_dir = experiment_root_dir.format(method)
        seed_dirs = glob(f"{experiment_root_dir}*")

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0,1,100)
        for i in range(args.k):
            fold_tprs = []
            fold_aucs = []
            fold_mean_fpr = np.linspace(0,1,100)
            for s, seed_dir in enumerate(seed_dirs):
                h5_file_dir = os.path.join(seed_dir, f"split_{i}_results/test_prob_label.h5")
                if not os.path.exists(h5_file_dir):
                    continue
                with h5py.File(h5_file_dir, 'r') as f:
                    labels = np.array(f['patient']['label'])
                    probs = np.array(f['patient']['prob'])
                
                fold_seed_fpr, fold_seed_tpr, _ = roc_curve(labels[:, gene_index], probs[:, gene_index])
                fold_tprs.append(interp(fold_mean_fpr, fold_seed_fpr, fold_seed_tpr))
                fold_seed_roc_auc = auc(fold_seed_fpr, fold_seed_tpr)
                fold_aucs.append(fold_seed_roc_auc)


            fold_mean_tpr = np.mean(fold_tprs, axis=0)
            fold_mean_auc = auc(fold_mean_fpr, fold_mean_tpr)
            fold_std_auc = np.std(fold_aucs)

            tprs.append(fold_mean_tpr)
            aucs.append(fold_mean_auc)

        mean_tpr = np.mean(tprs, axis = 0)
        mean_auc = auc(mean_fpr, mean_tpr)

        ccolor = next(color)
        plt.plot(mean_fpr, mean_tpr, color=ccolor,
                label=fr'{method} (AUC = %0.2f)' % (mean_auc),lw=2, alpha=1)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=ccolor,
            alpha=0.2
        )

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f' {gene.upper()} ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(f"{args.result_dir}/roc_{gene}.png")
    
    plt.clf()
            


def plot_pr_kfolds_nseeds(gene_index, gene):
    color = iter(cm.rainbow(np.linspace(0, 1, len(args.methods))))

    for method in args.methods:
        all_labels = []
        all_probs = []

        experiment_root_dir = args.experiment_root_dir
        experiment_root_dir = experiment_root_dir.format(method)
        seed_dirs = glob(f"{experiment_root_dir}*")

        prs = []
        aucs = []
        mean_recall = np.linspace(0,1,100)
        for i in range(args.k):
            fold_prs = []
            fold_aucs = []
            fold_mean_recall = np.linspace(0,1,100)
            for s, seed_dir in enumerate(seed_dirs):
                h5_file_dir = os.path.join(seed_dir, f"split_{i}_results/test_prob_label.h5")
                if not os.path.exists(h5_file_dir):
                    continue
                with h5py.File(h5_file_dir, 'r') as f:
                    labels = np.array(f['patient']['label'])
                    probs = np.array(f['patient']['prob'])
                    all_labels.append(labels[:, gene_index])
                    all_probs.append(probs[:, gene_index])

                
                fold_seed_pr, fold_seed_re, fold_seed_thresholds = precision_recall_curve(labels[:, gene_index], probs[:, gene_index])
                fold_prs.append(interp(fold_mean_recall, fold_seed_pr, fold_seed_re))
                fold_seed_pr_auc = auc(fold_seed_re, fold_seed_pr)
                fold_aucs.append(fold_seed_pr_auc)


            fold_mean_prs = np.mean(fold_prs, axis=0)
            fold_mean_auc = auc(fold_mean_recall, fold_mean_prs)
            fold_std_auc = np.std(fold_aucs)

            prs.append(fold_mean_prs)
            aucs.append(fold_mean_auc)

        mean_prs = np.mean(prs, axis = 0)
        mean_auc = auc(mean_recall, mean_prs)


        all_labels = np.concatenate(all_labels, axis = 0)
        all_probs = np.concatenate(all_probs, axis = 0)

        avg_pr = average_precision_score(all_labels, all_probs)

        ccolor = next(color)
        plt.plot(mean_recall, mean_prs, color=ccolor,
                label=fr'{method} (AUCPR = %0.2f, AP = %0.2f)' % (mean_auc, avg_pr),lw=2, alpha=1)

        # mean_prs = list(mean_prs)
        # mean_prs.reverse()
        # mean_prs = np.array(mean_prs)

        # mean_recall = list(mean_recall)
        # mean_recall.reverse()
        # mean_recall = np.array(mean_recall)

        std_prs = np.std(prs, axis=0)
        prs_upper = np.minimum(mean_prs + std_prs, 1)
        prs_lower = np.maximum(mean_prs - std_prs, 0)
        plt.fill_between(
            mean_recall,
            prs_lower,
            prs_upper,
            color=ccolor,
            alpha=0.2
        )

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f' {gene.upper()} PR Curve')
        plt.legend(loc="upper right")
        plt.savefig(f"{args.result_dir}/pr_{gene}.png")

    plt.clf()




if __name__ == "__main__":
    
    for i, gene in enumerate(args.gene):
        plot_pr_kfolds_nseeds(i, gene)
        plot_roc_kfolds_nseeds(i, gene)
