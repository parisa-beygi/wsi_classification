from cmath import isnan
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from models.resnet_custom import resnet50_baseline

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes (number of patches x n_classes)
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 

    instance-level training args:
    k_sample: number of ID highly-attended patch instances
    sample_number: total number of ID patch samples in each class queue for computing the distribution parameters
    k_sample_ood: number of OOD instance embedding samples for each class
    sample_from: number of embedding instances to sample from the distribution
    instance_loss_fn: loss function to differentiate ID and OOD data 
"""
class VOS_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False,  n_classes=2,
                k_sample=8, sample_number = 1000, k_sample_ood = 4, sample_from = 10000, instance_loss_fn=nn.CrossEntropyLoss(), start_epoch = 0):
        super(VOS_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.slide_classifier = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes

        self.k_sample = k_sample
        self.sample_number = sample_number
        self.k_sample_ood = k_sample_ood
        self.sample_from = sample_from
        self.instance_loss_fn = instance_loss_fn
        self.start_epoch = start_epoch
        
        # instance-level modules
        # maintaining a data queue for each class
        self.data_dict = torch.zeros(n_classes, sample_number, size[1]).cuda()
        # keep the track of the number of instances in the queue for each class  
        self.number_dict = {i: 0 for i in range(n_classes)}

        # for log_sum_exp computation
        self.weight_energy = torch.nn.Linear(n_classes, 1)
        torch.nn.init.uniform_(self.weight_energy.weight)


        # instance-level classifier
        # self.instance_classifier = nn.Linear(size[1], 2)

        # logistic regression for separating energies of ID and OOD data
        self.energy_regression = torch.nn.Linear(1, 2)


        initialize_weights(self)
        print ('here')

        # pretrained feature extractor model
        self.feature_extractor = resnet50_baseline(pretrained=True)


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.slide_classifier = self.slide_classifier.to(device)
        self.weight_energy = self.weight_energy.to(device)
        # self.instance_classifier = self.instance_classifier.to(device)
        self.energy_regression = self.energy_regression.to(device)
        self.feature_extractor = self.feature_extractor.to(device)



    def log_sum_exp(self, value, dim=None, keepdim=False):
        """
        Numerically stable implementation of the operation
        value.shape == [1, 2]
        value.exp().sum(dim, keepdim).log()
        
        """
        # TODO: torch.max(value, dim=None) threw an error at time of writing

        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True) ## m.shape == [128, 1]
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim) ## m.shape == [128]
            mult = F.relu(self.weight_energy.weight)
            sum_exp = torch.sum(
                mult * torch.exp(value0), dim=dim, keepdim=keepdim)

            res = m + torch.log(sum_exp + 1) ## F.relu.shape: torch.Size([1, 10]), value0.shape == [128, 10]
            return res
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)


    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        h_prev = h
        A, h = self.attention_net(h)  # NxK (N: number of patches in slide, K: number of classes in attention (=1)), h: N x 512        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        # instance evaluation:     
        if instance_eval:
            all_targets = []
            all_preds = []

            # (1) get the top attended (k_sample) patches in h
            top_p_ids = torch.topk(A, self.k_sample)[1][-1]
            top_p = torch.index_select(h, dim=0, index=top_p_ids)
            # (2) get the total number of samples in the class queues
            
            # (3) get the label
            label = label.item()

            # for evaluation data_dict is assumed to be complete
            # compute the mean for each class [2, 512], covariance [512, 512]
            for i in range(self.n_classes):
                if i == 0:
                    X = self.data_dict[i] - self.data_dict[i].mean(0) #deduct the mean of self.sample_number in class i from each emb in class i
                    mean_embed_id = self.data_dict[i].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, self.data_dict[i] - self.data_dict[i].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                            self.data_dict[i].mean(0).view(1, -1)), 0)
            ## X.shape == [2*self.sample_number, 512], mean_embed_id == [2, 512]
            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * torch.eye(X.shape[1], device=device)
            ## temp_precision.shape == [512, 512]

            # # debug:
            # mean_embed_id = mean_embed_id.detach()
            # temp_precision = temp_precision.detach()
            for index in range(self.n_classes):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                # new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                #     torch.zeros(mean_embed_id[index].shape).cuda(), covariance_matrix=torch.eye(temp_precision.shape[0]).cuda())

                negative_samples = new_dis.rsample((self.sample_from,)) #negative_samples.shape == [self.sample_from, 512]
                prob_density = new_dis.log_prob(negative_samples) ##prob_density.shape == [self.sample_from]
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(-prob_density, self.k_sample_ood)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
            ## ood_samples.shape == [2*self.k_sample_ood, 512]
            if len(ood_samples) != 0:
                # predictions_id = self.instance_classifier(top_p) # predictions_id: [self.k_sample, 2]
                predictions_id = self.slide_classifier(top_p) # predictions_id: [self.k_sample, 2]

                energy_score_for_fg = self.log_sum_exp(predictions_id, 1) ## energy_score_for_fg.shape == [self.k_sample]
                # debug:
                # predictions_ood = self.instance_classifier(ood_samples) ## predictions_ood.shape == [2*self.k_sample_ood, 2]
                predictions_ood = self.slide_classifier(ood_samples) ## predictions_ood.shape == [2*self.k_sample_ood, 2]

                # debug:
                energy_score_for_bg = self.log_sum_exp(predictions_ood, 1) ## energy_score_for_bg.shape == [2*self.k_sample_ood]
                # energy_score_for_bg = self.log_sum_exp(torch.rand(predictions_ood.shape).cuda(), 1) ## energy_score_for_bg.shape == [2*self.k_sample_ood]


                # debug:
                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1) # [self.k_sample + 2*self.k_sample_ood]
                # input_for_lr = torch.cat((energy_score_for_fg, torch.rand(energy_score_for_bg.shape).cuda()), -1) # [self.k_sample + 2*self.k_sample_ood]

                labels_for_lr = torch.cat((torch.ones(len(top_p)).cuda(),
                                        torch.zeros(len(ood_samples)).cuda()), -1) # [self.k_sample + 2*self.k_sample_ood]

                energy_pred = self.energy_regression(input_for_lr.view(-1, 1)) # energy_pred.shape == [self.k_sample + 2*self.k_sample_ood, 2]
                # debug:
                instance_loss = self.instance_loss_fn(energy_pred, labels_for_lr.long())
                # instance_loss = self.instance_loss_fn(torch.rand(energy_pred.shape).cuda(), labels_for_lr.long())


                # get instance preds
                instance_preds = torch.topk(energy_pred, 1, dim = 1)[1].squeeze(1)

                all_preds.extend(instance_preds.cpu().numpy())
                all_targets.extend(labels_for_lr.detach().cpu().numpy())

        M = torch.mm(A, h)  # M.shape == [1, 512]
        logits = self.slide_classifier(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': instance_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        # add embeddings before and after applying attention
        results_dict.update({'embeddings_before': h_prev})
        results_dict.update({'embeddings_after': h})


        return logits, Y_prob, Y_hat, A_raw, results_dict
            


    def forward_virtual(self, h, label=None, epoch = 0, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        h_prev = h
        A, h = self.attention_net(h)  # NxK (N: number of patches in slide, K: number of classes in attention (=1)), h: N x 512        


        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            all_targets = []
            all_preds = []

            instance_loss = torch.zeros(1).cuda()[0]

            # update the data_dict queue with the top attended (k_sample) patches (output) in this slide (h):
            # note that the queue only gets updated for class == label
            # (1) get the top attended (k_sample) patches in h
            top_p_ids = torch.topk(A, self.k_sample)[1][-1]
            top_p = torch.index_select(h, dim=0, index=top_p_ids)

            # (1-1) based on the coordinates of the top attended patches, forward the corresponding patch images to the feature extractor
            # then apply a linear transformation to reduce the diemnsions
            # then use those to update the queues
            sample_img = torch.rand(8, 3, 256, 256).to(device)
            feats = self.feature_extractor(sample_img)
            print (feats.shape)
            print (top_p.shape)
            



            # (2) get the total number of samples in the class queues
            sum_temp = 0
            for i in range(self.n_classes):
                sum_temp += self.number_dict[i]
            
            # (3) get the label
            label = label.item()


            if sum_temp == self.n_classes*self.sample_number and epoch < self.start_epoch:
                # if data_dict is complete but epoch still less than start_epoch
                if torch.isnan(top_p).any():
                    print ('here')
                self.data_dict[label] = torch.cat((self.data_dict[label][self.k_sample:], top_p.detach()), 0)

            elif sum_temp == self.n_classes*self.sample_number:
                # if the data_dict is complete and current epoch >= start_epoch
                if torch.isnan(top_p).any():
                    print ('here')
                self.data_dict[label] = torch.cat((self.data_dict[label][self.k_sample:], top_p.detach()), 0)

                # compute the mean for each class [2, 512], covariance [512, 512]
                for i in range(self.n_classes):
                    if i == 0:
                        X = self.data_dict[i] - self.data_dict[i].mean(0) #deduct the mean of self.sample_number in class i from each emb in class i
                        mean_embed_id = self.data_dict[i].mean(0).view(1, -1)
                    else:
                        X = torch.cat((X, self.data_dict[i] - self.data_dict[i].mean(0)), 0)
                        mean_embed_id = torch.cat((mean_embed_id,
                                                self.data_dict[i].mean(0).view(1, -1)), 0)
                ## X.shape == [2*self.sample_number, 512], mean_embed_id == [2, 512]
                ## add the variance.
                temp_precision = torch.mm(X.t(), X) / len(X)
                temp_precision += 0.0001 * torch.eye(X.shape[1], device=device)
                ## temp_precision.shape == [512, 512]

                # # debug:
                # mean_embed_id = mean_embed_id.detach()
                # temp_precision = temp_precision.detach()
                for index in range(self.n_classes):
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
                    # new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    #     torch.zeros(mean_embed_id[index].shape).cuda(), covariance_matrix=torch.eye(temp_precision.shape[0]).cuda())

                    negative_samples = new_dis.rsample((self.sample_from,)) #negative_samples.shape == [self.sample_from, 512]
                    prob_density = new_dis.log_prob(negative_samples) ##prob_density.shape == [self.sample_from]
                    # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                    # keep the data in the low density area.
                    cur_samples, index_prob = torch.topk(-prob_density, self.k_sample_ood)
                    if index == 0:
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                ## ood_samples.shape == [2*self.k_sample_ood, 512]
                if len(ood_samples) != 0:
                    # predictions_id = self.instance_classifier(top_p) # predictions_id: [self.k_sample, 2]
                    predictions_id = self.slide_classifier(top_p) # predictions_id: [self.k_sample, 2]

                    energy_score_for_fg = self.log_sum_exp(predictions_id, 1) ## energy_score_for_fg.shape == [self.k_sample]
                    # debug:
                    # predictions_ood = self.instance_classifier(ood_samples) ## predictions_ood.shape == [2*self.k_sample_ood, 2]
                    predictions_ood = self.slide_classifier(ood_samples) ## predictions_ood.shape == [2*self.k_sample_ood, 2]


                    # debug:
                    energy_score_for_bg = self.log_sum_exp(predictions_ood, 1) ## energy_score_for_bg.shape == [2*self.k_sample_ood]
                    # energy_score_for_bg = self.log_sum_exp(torch.rand(predictions_ood.shape).cuda(), 1) ## energy_score_for_bg.shape == [2*self.k_sample_ood]
                    if torch.isinf(energy_score_for_fg).any() or torch.isinf(energy_score_for_bg).any():
                        print ("Trouble in logsum!")


                    # debug:
                    input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1) # [self.k_sample + 2*self.k_sample_ood]
                    # input_for_lr = torch.cat((energy_score_for_fg, torch.rand(energy_score_for_bg.shape).cuda()), -1) # [self.k_sample + 2*self.k_sample_ood]

                    labels_for_lr = torch.cat((torch.ones(len(top_p)).cuda(),
                                            torch.zeros(len(ood_samples)).cuda()), -1) # [self.k_sample + 2*self.k_sample_ood]

                    energy_pred = self.energy_regression(input_for_lr.view(-1, 1)) # energy_pred.shape == [self.k_sample + 2*self.k_sample_ood, 2]
                    # debug:
                    instance_loss = self.instance_loss_fn(energy_pred, labels_for_lr.long())
                    # instance_loss = self.instance_loss_fn(torch.rand(energy_pred.shape).cuda(), labels_for_lr.long())


                    # get instance preds
                    instance_preds = torch.topk(energy_pred, 1, dim = 1)[1].squeeze(1)

                    all_preds.extend(instance_preds.cpu().numpy())
                    all_targets.extend(labels_for_lr.detach().cpu().numpy())

            else:
                sample_temp = self.number_dict[label]
                if sample_temp < self.sample_number:
                    num_add = min(self.sample_number - sample_temp, self.k_sample)
                    if torch.isnan(top_p[:num_add]).any():
                        print ('here')
                    self.data_dict[label][sample_temp: sample_temp + num_add] = top_p[:num_add].detach()
                    self.number_dict[label] += num_add
                else:
                    # in case the queue for class is full, dequeue old and enqueue new
                    if torch.isnan(top_p).any():
                        print ('here')                    
                    self.data_dict[label] = torch.cat((self.data_dict[label][self.k_sample:], top_p.detach()), 0)

            # for each class sample from Gaussian dist given the mean and cov of the class (given sample_from)
            # as a result: ood_samples.shape == [2, ood_n_samples, 512]
            
            # get the predictions for x and ood_samples
            # x_pred.shape == [k_sample, 2], ood_samples_pred.shape == [2*ood_n_samples, 2]
            # compute the energy for each of x_pred and ood_samples_pred: [k_sample], [2*ood_n_samples]
            # concat the energies: [k_sample + 2*ood_n_samples]
            # create id vs ood labels: [ones(k_sample), zeros(2*ood_n_samples)]
            # separate two eneries with a 2 to 1 classfier
            # compute instance-level loss
            
                

        M = torch.mm(A, h)  # M.shape == [1, 512]
        logits = self.slide_classifier(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': instance_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        # add embeddings before and after applying attention
        results_dict.update({'embeddings_before': h_prev})
        results_dict.update({'embeddings_after': h})

        return logits, Y_prob, Y_hat, A_raw, results_dict

