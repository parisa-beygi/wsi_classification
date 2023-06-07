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

        self.n_labels = n_labels


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention = self.attention.to(device)


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
        A = self.attention(x)
        for i in range(self.n_labels):
            A[:, i] = A[:, i].masked_fill((x == 0).all(dim=1).reshape(A[:, i].shape), -9e15) # filter padded rows

        return A, x
    
    def compute_agg(self, A, x):
        k, c = x.shape
        M = torch.einsum('d k, k c -> d c', A, x)  # d, c
        x_ex = einops.repeat(x, 'k c -> d k c', d=self.n_labels)
        
        S = torch.pow(x_ex - M.reshape(self.n_labels, 1, c), 2) # d, k, c
        V = torch.einsum('d k, d k c -> d c', A, S)
        nb_patch = (torch.tensor(k).expand(self.n_labels)).to(self.device)
        nb_patch = nb_patch - torch.sum((x == 0).all(dim=1), dim=0)    # filter padded rows
        nb_patch = nb_patch / (nb_patch - 1)                           # I / I-1
        nb_patch = torch.nan_to_num(nb_patch, posinf=1)                # for cases, when we have only 1 patch (inf)
        V = V * nb_patch[:, None]                                      # broadcasting
        M_V = torch.cat((M, V), dim=1)

        return M_V

class Attention(nn.Module):
    def __init__(self, L = 1024, D = 128, n_labels = 4, n_classes = 2):
        super(Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = n_labels

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention = self.attention.to(device)


    def forward(self, x):
        x = x.squeeze(0)
        
        A = self.attention(x)  # NxK

        return A, x

    def compute_agg(self, A, x):
        return torch.mm(A, x)

class GatedAttention(nn.Module):
    def __init__(self, L = 1024, D = 128, n_labels = 4, n_classes = 2):
        super(GatedAttention, self).__init__()
        self.L = L
        self.D = D
        self.K = n_labels

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.attention_V = self.attention_V.to(device)
        self.attention_U = self.attention_U.to(device)
        self.attention_weights = self.attention_weights.to(device)

    def forward(self, x):
        x = x.squeeze(0)

        H = x

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK

        return A, x


    def compute_agg(self, A, x):
        return torch.mm(A, x)