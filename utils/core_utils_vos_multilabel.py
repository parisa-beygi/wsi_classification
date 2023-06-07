from unittest import TestResult
import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_vos_multilabel import VOS_MB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import pandas as pd
import functools


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_balanced_acc(self):
        sum = 0.
        for c in range(self.n_classes):
            acc, _, _ = self.get_summary(c)
            sum += acc
        return sum / self.n_classes

    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class Accuracy_Logger_Multilabel(object):
    """Accuracy logger"""
    def __init__(self, n_classes, n_labels):
        super(Accuracy_Logger_Multilabel, self).__init__()
        self.n_classes = n_classes
        self.n_labels = n_labels
        self.initialize()

    def initialize(self):
        self.data = [[{"count": 0, "correct": 0} for i in range(self.n_classes)] for j in range(self.n_labels)]
    
    def log(self, Y_hat, Y):
        for label in range(Y.shape[-1]):
            L_Y_hat = int(Y_hat[:, label])
            L_Y = int(Y[:, label])
            self.data[label][L_Y]["count"] += 1
            self.data[label][L_Y]["correct"] += (L_Y_hat == L_Y)
        
    def get_balanced_acc(self, label):
        sum = 0.
        for c in range(self.n_classes):
            acc, _, _ = self.get_summary(label, c)
            sum += acc
        return sum / self.n_classes

    def get_summary(self, label, c):
        count = self.data[label][c]["count"] 
        correct = self.data[label][c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count


class PerformanceEarlyStopping:
    """Early stops the training if a performance measure doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_max = -np.Inf

    def __call__(self, epoch, score, model, ckpt_name = 'checkpoint.pt'):
        # the larger score the better
        score = score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, score, model, ckpt_name):
        '''Saves model when score increases.'''
        if self.verbose:
            print(f'Score increased ({self.score_max:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.score_max = score


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

# import pandas as pd
# def train_pooch(datasets, cur, args):
#     test_acc_data = {'acc': [11, 22], 'correct': [13, 24], 'count': [15, 26]}
#     val_acc_data = {'acc': [17, 28], 'correct': [19, 20], 'count': [11, 22]}

#     return None, 1, 2, 3, 4, pd.DataFrame(test_acc_data), pd.DataFrame(val_acc_data)

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    train_weights = get_class_weights_for_multilabel_loss(train_split)
    loss_fn = functools.partial(multilabel_CE, train_weights)
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'n_labels': args.n_labels}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb_vos', 'clam_mb_vos']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.start_epoch:
            model_dict.update({"start_epoch": args.start_epoch})
        if args.sample_number is not None:
            model_dict.update({"sample_number": args.sample_number})
        if args.k_sample_ood is not None:
            model_dict.update({"k_sample_ood": args.k_sample_ood})
        if args.sample_from is not None:
            model_dict.update({"sample_from": args.sample_from})            
        
        if args.model_type =='clam_sb_vos':
            model = VOS_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb_vos':
            model = VOS_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader_multilabel(train_split, training=True, testing = args.testing)
    val_loader = get_split_loader_multilabel(val_split,  testing = args.testing)
    test_loader = get_split_loader_multilabel(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = PerformanceEarlyStopping(patience = 20, stop_epoch=50, verbose = True)
        # early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb_vos', 'clam_mb_vos'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.n_labels, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, args.n_labels,
                early_stopping, writer, loss_fn, args.results_dir)
        
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, args.n_labels, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, args.n_labels,
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))


    val_results_label_list, val_prob_label_dict_label_list = prepare_reports(model, val_loader, args.n_classes, args.labels, group = 'val', writer = writer)
    test_results_label_list, test_prob_label_dict_label_list = prepare_reports(model, test_loader, args.n_classes, args.labels, group = 'test', writer = writer)

    split_counts_list = [get_split_counts(lable_j, label_name, args.n_classes, train = train_split, val = val_split, test = test_split) for lable_j, label_name in enumerate(args.labels)]

    # print ('*** Val logs ***')
    # val_patient_results, val_error, val_auc, val_acc_logger= summary(model, val_loader, args.n_classes)
    # print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    # val_acc_data = {'acc': [], 'correct': [], 'count': []}
    # for i in range(args.n_classes):
    #     acc, correct, count = val_acc_logger.get_summary(i)
    #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    #     val_acc_data['acc'].append(acc)
    #     val_acc_data['correct'].append(correct)
    #     val_acc_data['count'].append(count)

    #     if writer:
    #         writer.add_scalar('final/val_class_{}_acc'.format(i), acc, 0)

    # print ('*** Test logs ***')
    # results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    # print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    # test_acc_data = {'acc': [], 'correct': [], 'count': []}
    # for i in range(args.n_classes):
    #     acc, correct, count = acc_logger.get_summary(i)
    #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    #     test_acc_data['acc'].append(acc)
    #     test_acc_data['correct'].append(correct)
    #     test_acc_data['count'].append(count)

    #     if writer:
    #         writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    # if writer:
    #     writer.add_scalar('final/val_error', val_error, 0)
    #     writer.add_scalar('final/val_auc', val_auc, 0)
    #     writer.add_scalar('final/test_error', test_error, 0)
    #     writer.add_scalar('final/test_auc', test_auc, 0)
    #     writer.close()
    # return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, pd.DataFrame(test_acc_data), pd.DataFrame(val_acc_data)

    if writer:
        writer.close()
    
    return val_results_label_list, test_results_label_list, val_prob_label_dict_label_list, test_prob_label_dict_label_list, split_counts_list


def get_split_counts(label, label_name, n_classes, **kwargs):
    count_data = np.zeros((2*len(kwargs), n_classes+1))
    rows = []
    cols = [f'class_{i}' for i in range(n_classes)]
    cols.append('total')
    for i, (key, split) in enumerate(kwargs.items()):
        index = 2*i
        for j in range(n_classes):
            count_data[index, j] = len(split.slide_cls_ids[label][j])
            count_data[index+1, j] = len(split.patient_cls_ids[label][j])
        count_data[index, -1] = np.sum(count_data[index, :-1])
        count_data[index+1, -1] = np.sum(count_data[index+1, :-1])

        rows.append(f'{key}_slide_{label_name}')
        rows.append(f'{key}_patient_{label_name}')

    
    I = pd.Index(rows)
    C = pd.Index(cols)

    return pd.DataFrame(data=count_data, index=I, columns=C)

def get_results_data(level, group, n_classes, labels, error, auc, acc_logger, writer):
    """
    data: each element is level_group_jlabel for the table with columns acc_class_0, acc_class_1, balanced_acc, auc
    """
    data = []
    for j in range(len(labels)):
        values = []
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(j, i)
            print('{} {} - label {} - class {}: acc {}, correct {}/{}'.format(group, level, labels[j], i, acc, correct, count))
            values.append(acc)
            if writer:
                writer.add_scalar(f'final/{group}_label_{labels[j]}_class_{i}_acc_{level}', acc, 0)


        balanced_acc = acc_logger.get_balanced_acc(j)
        values.append(balanced_acc)
        values.append(auc[j])

        if writer:
            writer.add_scalar(f'final/{group}_error_{level}_label_{labels[j]}', error, 0)
            writer.add_scalar(f'final/{group}_auc_{level}_label_{labels[j]}', auc[j], 0)

        data.append(values)
        
    data = np.array(data)
    
    return data


def prepare_reports(model, loader, n_classes, labels, group = 'test', writer = None):
    n_labels = len(labels)
    patient_results, error, auc, acc_logger, slide_probs, slide_labels = summary(model, loader, n_classes, n_labels)
    prob_label_dict_label = {'slide': {'prob': slide_probs, 'label': slide_labels}}


    metrics = [f'{group}_class_{i}_acc' for i in range(n_classes)]
    metrics.append(f"{group}_balanced_acc")
    metrics.append(f"{group}_auc")

    # metrics = list(map(lambda x: f"{group}_{x}", metrics))
    I = pd.Index([f'{level}_label_{labels[j]}' for level in ["slide_level", "patient_level"] for j in range(n_labels)])
    C = pd.Index(metrics)

    agg_error, agg_auc, agg_acc_logger, agg_probs, agg_labels = aggregate_results(patient_results, n_classes=n_classes, n_labels=n_labels)
    prob_label_dict_label.update({"patient": {'prob': agg_probs, 'label': agg_labels}})

    slide_data = get_results_data('slide', group, n_classes, labels, error, auc, acc_logger, writer)    
    patient_data = get_results_data('patient', group, n_classes, labels, agg_error, agg_auc, agg_acc_logger, writer)
           
    data = np.concatenate((slide_data, patient_data))
    df = pd.DataFrame(data=data, index=I, columns=C)

    return df, prob_label_dict_label



def train_loop_clam(epoch, model, loader, optimizer, n_classes, n_labels, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger_Multilabel(n_classes=n_classes, n_labels=n_labels)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    inst_preds = []
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model.forward_virtual(data, labels=label, epoch = epoch, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 


        ## for debugging
        prev_inst_preds = inst_preds


        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        if len(prev_inst_preds) == 0 and len(inst_preds) != 0:
            print (f'begining of issue at batch idx {batch_idx}!')


        train_loss += loss_value
        if epoch >= 35 and batch_idx >= 159:
            print (f'batch {batch_idx}')

        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.cpu().detach().numpy(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for j in range(n_labels):
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(j, i)
            print('label {} - class {}: acc {}, correct {}/{}'.format(j, i, acc, correct, count))
            if writer and acc is not None:
                writer.add_scalar('train/label_{}_class_{}_acc'.format(j, i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, n_labels, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger_Multilabel(n_classes=n_classes, n_labels=n_labels)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, n_labels, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger_Multilabel(n_classes=n_classes, n_labels=n_labels)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        balanced_acc = acc_logger.get_balanced_acc()
        early_stopping(epoch, balanced_acc, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        # early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, n_labels, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger_Multilabel(n_classes=n_classes, n_labels=n_labels)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_labels))
    labels = np.zeros((len(loader), n_labels))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, labels=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.cpu().numpy()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        aucs = []
        for j in range(n_labels):
            auc = roc_auc_score(labels[:, j], prob[:, j])
            aucs.append(auc)
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}'.format(val_loss, val_error))
    for j in range(n_labels):
        print('\nLabel {} auc: {:.4f}'.format(j, aucs[j]))

    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
        for j in range(n_labels):
            writer.add_scalar('val/auc/label_{}'.format(j), aucs[j], epoch)

    for j in range(n_labels):
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(j, i)
            print('label {} - class {}: acc {}, correct {}/{}'.format(j, i, acc, correct, count))
            
            if writer and acc is not None:
                writer.add_scalar('val/label_{}_class_{}_acc'.format(j, i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes, n_labels):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger_Multilabel(n_classes=n_classes, n_labels=n_labels)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_labels))
    all_labels = np.zeros((len(loader), n_labels))

    slide_ids = loader.dataset.slide_data[['slide_id', 'case_id']]
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id, case_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.cpu().numpy()


        # patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        if case_id not in patient_results:
            patient_results[case_id] = {}
        patient_results[case_id].update({slide_id: {'prob': probs, 'label': label.cpu().numpy()}})

        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        aucs = []
        for j in range(n_labels):
            auc = roc_auc_score(all_labels[:, j], all_probs[:, j])
            aucs.append(auc)
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, aucs, acc_logger, all_probs, all_labels


def aggregate_results(results_dict, n_classes = 2, n_labels = 2, aggregate_level = 'patient'):
    """
    results_dic: {'case_id': {'slide_id': {'prob': np_array, 'label': int}}}

    """
    acc_logger = Accuracy_Logger_Multilabel(n_classes=n_classes, n_labels=n_labels)

    all_probs = np.zeros((len(results_dict), n_labels))
    all_labels = np.zeros((len(results_dict), n_labels))

    total_error = 0.
    for i, case_id in enumerate(results_dict):
        case_probs = []
        case_labels = []
        for slide_id in results_dict[case_id]:
            case_probs.append(results_dict[case_id][slide_id]['prob'])
            case_labels.append(results_dict[case_id][slide_id]['label'])
        
        case_probs = np.stack(case_probs, axis=0)

        # # how to aggregate slides probabilities for a patient
        # # get the mean for each class across all the slides
        # prob = np.mean(case_probs, axis=0)
        prob = np.mean(case_probs, axis = 0)

        Y_hat = np.round(prob)
        # assumed the labels are patient-level and
        # all slides from a given patient share the same label
        # that is why we want to aggregate slides predictions into patient level's 
        label = case_labels[0]

        acc_logger.log(Y_hat, label)


        all_probs[i] = prob
        all_labels[i] = label

        error = calculate_error(torch.Tensor(Y_hat), torch.Tensor([label]))
        total_error += error

    total_error /= len(results_dict)

    if n_classes == 2:
        aucs = []
        for j in range(n_labels):
            auc = roc_auc_score(all_labels[:, j], all_probs[:, j])
            aucs.append(auc)
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    # return aucs instead of auc, because of multi-label binary classification

    return total_error, aucs, acc_logger, all_probs, all_labels
