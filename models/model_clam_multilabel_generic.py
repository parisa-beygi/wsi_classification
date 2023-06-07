import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from models.attention_models import clam, ABMIL, local_vit


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, att_net = 'clam_gated', size_arg = "small", dropout = False, k_sample=8, n_classes=2, n_labels = 2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        classifier_input_size = size[1]
        if att_net == 'clam_gated':
            self.attention_net = clam.Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        elif att_net == 'clam':
            self.attention_net = clam.Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        elif att_net == 'ABMIL':
            self.attention_net = ABMIL.Attention(L = size[1], n_labels = 1)
        elif att_net == 'ABMIL_gated':
            self.attention_net = ABMIL.GatedAttention(L = size[1], n_labels = 1)
        elif att_net == 'VarMIL':
            classifier_input_size = 2*size[1]
            self.attention_net = ABMIL.VarMIL(L = size[1], n_labels = 1)
        elif att_net == 'LocalViT':
            self.attention_net = local_vit.LocalViT(L = size[0], D = size[1], n_classes = n_labels)
        

        self.att_net_type = att_net

        self.first_fc = nn.Sequential(*fc)

        bag_classifiers = [nn.Linear(classifier_input_size, 1) for i in range(n_labels)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        # label_classifiers = [nn.Linear(size[1], 1) for i in range(n_labels)] #use an indepdent linear layer to predict each class
        # self.classifiers = nn.ModuleList(label_classifiers)
        # for each label, there are two linears one for class 0 and one for class 1
        instance_classifiers = [[nn.Linear(size[1], 2) for i in range(n_classes)] for j in range(n_labels)]
        self.instance_classifiers = [nn.ModuleList(instance_classifier) for instance_classifier in instance_classifiers]
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.n_labels = n_labels
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.first_fc = self.first_fc.to(device)
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = [instance_classifier.to(device) for instance_classifier in self.instance_classifiers]
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.att_net_type == 'LocalViT':
            A, h = self.attention_net(h[0], h[1])  # NxK (N: number of patches in slide, K: number of classes in attention (=1)), h: N x 512 
            A = A.repeat(1, self.n_labels)
        else:
            x = self.first_fc(h)
            A, h = self.attention_net(x)  # NxK (N: number of patches in slide, K: number of classes in attention (=1)), h: N x 512        

        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for j in range(self.n_labels):
                for i in range(self.n_classes):
                    inst_label = inst_labels[j][i].item()
                    classifier = self.instance_classifiers[j][i]
                    if inst_label == 1: #in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else: #out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= (self.n_labels * self.n_classes)
            else:
                total_inst_loss /= self.n_labels

                
        # M = torch.mm(A, h)  # M.shape == [1, 512]
        M = self.attention_net.compute_agg(A, h) # n_labels, 512
        logits = torch.empty(1, self.n_labels).float().to(device)
        for c in range(self.n_labels):
            logits[:, c] = self.classifiers[c](M[0])
        Y_prob = F.sigmoid(logits)
        Y_hat = torch.round(Y_prob)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(self, att_net = 'clam_gated', size_arg = "small", dropout = False, k_sample=8, n_classes=2, n_labels = 2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        
        classifier_input_size = size[1]
        if att_net == 'clam_gated':
            self.attention_net = clam.Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_labels)
        elif att_net == 'clam':
            self.attention_net = clam.Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_labels)
        elif att_net == 'ABMIL':
            self.attention_net = ABMIL.Attention(L = size[1], n_labels = n_labels)
        elif att_net == 'ABMIL_gated':
            self.attention_net = ABMIL.GatedAttention(L = size[1], n_labels = n_labels)
        elif att_net == 'VarMIL':
            classifier_input_size = 2*size[1]
            self.attention_net = ABMIL.VarMIL(L = size[1], n_labels = n_labels)
        elif att_net == 'LocalViT':
            self.attention_net = local_vit.LocalViT(L = size[0], D = size[1], n_classes = n_labels)
        

        self.att_net_type = att_net

        

        # fc.append(self.att_subnet)
        self.first_fc = nn.Sequential(*fc)

        bag_classifiers = [nn.Linear(classifier_input_size, 1) for i in range(n_labels)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        self.n_labels = n_labels
        self.n_classes = n_classes

        instance_classifiers = [[nn.Linear(size[1], 2) for i in range(n_classes)] for j in range(n_labels)]
        self.instance_classifiers = [nn.ModuleList(instance_classifier) for instance_classifier in instance_classifiers]

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.first_fc = self.first_fc.to(device)
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = [instance_classifier.to(device) for instance_classifier in self.instance_classifiers]

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.att_net_type == 'LocalViT':
            A, h = self.attention_net(h[0], h[1])  # NxK (N: number of patches in slide, K: number of classes in attention (=1)), h: N x 512 
            A = A.repeat(1, self.n_labels)
        else:
            x = self.first_fc(h)
            A, h = self.attention_net(x)  # NxK (N: number of patches in slide, K: number of classes in attention (=1)), h: N x 512        

        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for j in range(self.n_labels):
                for i in range(self.n_classes):
                    inst_label = inst_labels[j][i].item()
                    classifier = self.instance_classifiers[j][i]
                    if inst_label == 1: #in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A[j], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else: #out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A[j], h, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= (self.n_labels * self.n_classes)
            else:
                total_inst_loss /= self.n_labels

        # M = torch.mm(A, h) 
        M = self.attention_net.compute_agg(A, h) # n_labels, 512
        logits = torch.empty(1, self.n_labels).float().to(device)
        for c in range(self.n_labels):
            m_index = 0 if self.att_net_type == 'LocalViT' else c # for local_vit, because it uses only one representation for all labels
            logits[:, c] = self.classifiers[c](M[m_index])
        Y_prob = torch.sigmoid(logits)
        Y_hat = torch.round(Y_prob)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
