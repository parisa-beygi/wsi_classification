import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class VarMIL(nn.Module):
    def __init__(self, L = 1024, D = 128, n_labels = 4, n_classes = 2):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attention  = nn.Sequential(nn.Linear(L, D),
                                       nn.Tanh(),
                                       nn.Linear(D, n_labels))
        bag_classifiers = [nn.Sequential(nn.Linear(2*L, D),
                                       nn.ReLU(),
                                       nn.Linear(D, 1)) for i in range(n_labels)]
        self.classifiers = nn.ModuleList(bag_classifiers)

        self.n_labels = n_labels


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention = self.attention.to(device)
        self.classifiers = self.classifiers.to(device)


    def forward(self, x):
        """
        x   (input)            :  K (nb_patch) x out_channel
        A   (attention weights):  K (nb_patch) x 1
        M   (weighted mean)    :  out_channel
        S   (std)              :  K (nb_patch) x out_channel
        V   (weighted variance):  out_channel
        nb_patch (nb of patch) : 
        M_V (concate M and V)  :  2*out_channel
        out (final output)     :  num_classes
        """
        k, c = x.shape
        A = self.attention(x)
        for i in range(self.n_labels):
            A[:, i] = A[:, i].masked_fill((x == 0).all(dim=1).reshape(A[:, i].shape), -9e15) # filter padded rows
        A = F.softmax(A, dim=0)                                        # softmax over K
        M = torch.einsum('k d, k c -> d c', A, x)  # d, c
        x_ex = einops.repeat(x, 'k c -> d k c', d=self.n_labels)
        
        S = torch.pow(x_ex - M.reshape(self.n_labels, 1, c), 2) # d, k, c
        V = torch.einsum('k d, d k c -> d c', A, S)
        nb_patch = (torch.tensor(k).expand(self.n_labels)).to(self.device)
        nb_patch = nb_patch - torch.sum((x == 0).all(dim=1), dim=0)    # filter padded rows
        nb_patch = nb_patch / (nb_patch - 1)                           # I / I-1
        nb_patch = torch.nan_to_num(nb_patch, posinf=1)                # for cases, when we have only 1 patch (inf)
        V = V * nb_patch[:, None]                                      # broadcasting
        M_V = torch.cat((M, V), dim=1)

        logits = torch.empty(1, self.n_labels).float().to(self.device)
        for c in range(self.n_labels):
            logits[:, c] = self.classifiers[c](M_V[c])

        Y_prob = torch.sigmoid(logits)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return logits, Y_prob, Y_hat
