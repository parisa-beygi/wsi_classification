import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_baseline import FC
from models.model_baseline import Kather
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import pandas as pd

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
        # the larger score the better
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

def train(datasets, cur, args):
    """   
        train patch-level for a single fold
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
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    if args.model_type == 'baseline':
        model_dict = {"model_size": args.model_size, "dropout": args.drop_out, 'n_classes': args.n_classes}
        model = FC(**model_dict)
    elif args.model_type == 'baseline_finetune':
        model_dict = {'n_classes': args.n_classes, 'feature_extracting': False}
        model = Kather(**model_dict)
    else:
        raise NotImplementedError
        

    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, batch_size=args.batch_size, collate='simple')
    val_loader = get_split_loader(val_split,  testing = args.testing, batch_size=args.batch_size, collate='simple')
    test_loader = get_split_loader(test_split, testing = args.testing, batch_size=args.batch_size, collate='simple')
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = PerformanceEarlyStopping(patience = 20, stop_epoch=50, verbose = True)
        # early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)


    else:
        early_stopping = None
    print('Done!')


    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))


    val_results, val_prob_label_dict = prepare_reports(model, val_loader, args.n_classes, group = 'val', writer = writer)
    test_results, test_prob_label_dict = prepare_reports(model, test_loader, args.n_classes, group = 'test', writer = writer)

    split_counts = get_split_counts(args.n_classes, train = train_split, val = val_split, test = test_split)


    if writer:
        writer.close()
    
    return val_results, test_results, val_prob_label_dict, test_prob_label_dict, split_counts



def get_split_counts(n_classes, **kwargs):
    count_data = np.zeros((3*len(kwargs), n_classes+1))
    rows = []
    cols = [f'class_{i}' for i in range(n_classes)]
    cols.append('total')
    for i, (key, split) in enumerate(kwargs.items()):
        index = 3*i
        for j in range(n_classes):
            count_data[index, j] = split.num_patches*len(split.slide_cls_ids[j])
            count_data[index+1, j] = len(split.slide_cls_ids[j])
            count_data[index+2, j] = len(split.patient_cls_ids[j])
        count_data[index, -1] = np.sum(count_data[index, :-1])
        count_data[index+1, -1] = np.sum(count_data[index+1, :-1])
        count_data[index+2, -1] = np.sum(count_data[index+2, :-1])


        rows.append(f'{key}_patch')
        rows.append(f'{key}_slide')
        rows.append(f'{key}_patient')

    
    I = pd.Index(rows)
    C = pd.Index(cols)

    return pd.DataFrame(data=count_data, index=I, columns=C)



def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = []
    all_labels = []

    batch_size = loader.batch_size
    patch_data = loader.dataset.patch_data[['slide_id', 'case_id']]
    patient_results = {}


    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        # slide_id, case_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _ = model(data)

        Y_hat, label = Y_hat.cpu(), label.cpu()        
        error = calculate_error(Y_hat, label)
        test_error += error

        acc_logger.log_batch(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        Y_hat, label = Y_hat.numpy(), label.numpy()        

        all_probs.extend(probs)
        all_labels.extend(label)
        



        last_index = (batch_idx + 1)*batch_size if (batch_idx + 1) < len(loader) else len(patch_data)
        slice_patch_data = patch_data.iloc[batch_idx*batch_size: last_index, :]
        for i, (slide_id, case_id) in enumerate(np.array(slice_patch_data)):
            if case_id not in patient_results:
                patient_results[case_id] = {}
            if slide_id not in patient_results[case_id]:
                patient_results[case_id][slide_id] = {'prob': [], 'label': []}
            patient_results[case_id][slide_id]['prob'].append(probs[i])
            patient_results[case_id][slide_id]['label'].append(label[i])

        # if (batch_idx + 1) % 20 == 0:
        #     print ('breaking for debugging ...')
        #     break


    test_error /= len(loader)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
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


    return patient_results, test_error, auc, acc_logger, all_probs, all_labels


def aggregate_results(results_dict, n_classes = 2, aggregate_level = 'patient'):
    """
    results_dic: {'case_id': {'slide_id': {'prob': [np_array], 'label': [int]}}}

    """
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # all_probs = np.zeros((len(results_dict), n_classes))
    # all_labels = np.zeros(len(results_dict))

    all_probs = []
    all_labels = []

    total_error = 0.
    for i, case_id in enumerate(results_dict):
        case_probs = []
        case_labels = []
        for slide_id in results_dict[case_id]:
            if aggregate_level == 'slide':
                slide_probs = np.array(results_dict[case_id][slide_id]['prob'])
                slide_labels = np.array(results_dict[case_id][slide_id]['label'])

                slide_hats = np.argmax(slide_probs, axis=1)
                prob = np.bincount(slide_hats, minlength=n_classes) / len(slide_hats)
                Y_hat = np.argmax(prob)
                label = slide_labels[0]
                acc_logger.log(Y_hat, label)

                all_probs.append(prob)
                all_labels.append(label)

                error = calculate_error(torch.Tensor(Y_hat), torch.Tensor([label]))
                total_error += error
            
            elif aggregate_level == 'patient':
                case_probs.extend(results_dict[case_id][slide_id]['prob'])
                case_labels.extend(results_dict[case_id][slide_id]['label'])

        if aggregate_level == 'patient':
            case_probs = np.array(case_probs)
            case_labels = np.array(case_labels)

            case_hats = np.argmax(case_probs, axis=1)
            prob = np.bincount(case_hats, minlength=n_classes) / len(case_hats)
            Y_hat = np.argmax(prob)

            # assumed the labels are patient-level and
            # all slides from a given patient share the same label
            # that is why we want to aggregate slides predictions into patient level's 
            label = case_labels[0]

            acc_logger.log(Y_hat, label)

            all_probs.append(prob)
            all_labels.append(label)

            error = calculate_error(torch.Tensor(Y_hat), torch.Tensor([label]))
            total_error += error

    total_error /= len(results_dict)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)


    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
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




    return total_error, auc, acc_logger, all_probs, all_labels



def prepare_reports(model, loader, n_classes, group = 'test', writer = None):
    patient_results, error, auc, acc_logger, patch_probs, patch_labels = summary(model, loader, n_classes)

    prob_label_dict = {'patch': {'prob': patch_probs, 'label': patch_labels}}

    metrics = [f'{group}_class_{i}_acc' for i in range(n_classes)]
    metrics.append(f"{group}_balanced_acc")
    metrics.append(f"{group}_auc")

    # metrics = list(map(lambda x: f"{group}_{x}", metrics))
    I = pd.Index(["patch_level", "slide_level", "patient_level"])
    C = pd.Index(metrics)


    data = []
    for level in ["patch", "slide", "patient"]:
        if level != "patch":
            error, auc, acc_logger, agg_probs, agg_labels = aggregate_results(patient_results, n_classes=n_classes, aggregate_level=level)
            prob_label_dict.update({level: {'prob': agg_probs, 'label': agg_labels}})
        values = []
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('{} {} - class {}: acc {}, correct {}/{}'.format(group, level, i, acc, correct, count))
            values.append(acc)
            if writer:
                writer.add_scalar(f'final/{group}_class_{i}_acc_{level}', acc, 0)

        balanced_acc = acc_logger.get_balanced_acc()
        values.append(balanced_acc)
        values.append(auc)

        if writer:
            writer.add_scalar(f'final/{group}_error_{level}', error, 0)
            writer.add_scalar(f'final/{group}_auc_{level}', auc, 0)


        data.append(values)
    
    data = np.array(data)


    return pd.DataFrame(data=data, index=I, columns=C), prob_label_dict





def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, y_probs, y_hats, _ = model(data)
        
        logits, y_hats, label = logits.cpu(), y_hats.cpu(), label.cpu()
        acc_logger.log_batch(y_hats, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}'.format(batch_idx, loss_value))
            # print ('breaking for debugging ...')
            # break
           
        error = calculate_error(y_hats, label)
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


def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    # prob = np.zeros((len(loader), n_classes))
    # labels = np.zeros(len(loader))

    prob = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _ = model(data)

            logits, Y_hat, label = logits.cpu(), Y_hat.cpu(), label.cpu()
            error = calculate_error(Y_hat, label)
            val_error += error

            acc_logger.log_batch(Y_hat, label)            
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            Y_prob = Y_prob.cpu().numpy()
            Y_hat, label = Y_hat.numpy(), label.numpy()        

            prob.extend(Y_prob)
            labels.extend(label)


            
            

    val_error /= len(loader)
    val_loss /= len(loader)

    prob = np.array(prob)
    labels = np.array(labels)

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


