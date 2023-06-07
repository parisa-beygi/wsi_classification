import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

from utils.gpu_manager import free_gpu_cache

# from kmeans_pytorch import kmeans
from fast_pytorch_kmeans import KMeans


class MultiHeadClusterAttention(nn.Module):
    def __init__(self, cluster_coeff = 100, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.cluster_coeff = cluster_coeff
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, A: Tensor, mask: Tensor = None) -> Tensor:
        # A: represents the kNN graph with shape (1, n, k) indicating k neighbors for all n patches
        # split keys, queries and values in num_heads
        
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        b, heads, n, d = queries.shape

        num_clusters = n // self.cluster_coeff
        if num_clusters > 10:
            kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', verbose=0)
            kmeans.fit_predict(x[0, :, :])
            x = kmeans.centroids.unsqueeze(0)

        
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int = 1024, target_emb_size: int = 512, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, target_emb_size),
            nn.ReLU(),
            nn.Dropout(drop_p)
        )

class ClusterTransformerEncoder(nn.Module):
    def __init__(self,
                 cluster_coeff_list = [100, 50],
                 input_emb_size: int = 1024,
                 emb_size: int = 512,
                 drop_p: float = 0.,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super(ClusterTransformerEncoder, self).__init__()

        # super().__init__(
        #     FeedForwardBlock(emb_size=input_emb_size, target_emb_size=emb_size, drop_p=forward_drop_p),
        #     ResidualAdd(nn.Sequential(
        #         MultiHeadClusterAttention(emb_size, **kwargs),
        #         nn.Dropout(drop_p)
        #     )),
        #     nn.LayerNorm(emb_size),
        #     ResidualAdd(nn.Sequential(
        #         MultiHeadClusterAttention(emb_size, **kwargs),
        #         nn.Dropout(drop_p)
        #     )),
        #     nn.LayerNorm(emb_size)
        #     )


        self.fc = FeedForwardBlock(emb_size=input_emb_size, target_emb_size=emb_size, drop_p=forward_drop_p)
        self.mhla_1 = MultiHeadClusterAttention(cluster_coeff_list[0], emb_size, **kwargs)
        self.ln_1 = nn.LayerNorm(emb_size)

        self.mhla_2 = MultiHeadClusterAttention(cluster_coeff_list[1], emb_size, **kwargs)
        self.ln_2 = nn.LayerNorm(emb_size)

        self.dropout = drop_p

    def forward(self, x, A, **kwargs):
        embs = self.fc(x)
        block_1 = self.mhla_1(embs, A[:, :, :16], **kwargs)
        block_1 = F.dropout(block_1, self.dropout)
        block_1 += embs
        block_1 = self.ln_1(block_1)

        block_2 = self.mhla_2(block_1, A[:, :, :64], **kwargs)
        block_2 = F.dropout(block_2, self.dropout)
        block_2 += block_1
        block_2 = self.ln_2(block_2)

        return block_2




class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 512, n_classes: int = 2):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))



class ClusterViT(nn.Module):
    def __init__(self,
                cluster_coeff_list = [100, 50],     
                input_emb_size: int = 1024,
                emb_size: int = 512,
                n_classes: int = 2,
                **kwargs):
        super(ClusterViT, self).__init__()
        self.encoder = ClusterTransformerEncoder(cluster_coeff_list=cluster_coeff_list, input_emb_size=input_emb_size, emb_size=emb_size, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.encoder.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, x, A):
        embs = self.encoder(x, A)
        if torch.isnan(embs).any():
            fake_embs = self.encoder(x, A)
        logits = self.classifier(embs)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat

# x = torch.rand(1, 10000, 1024)
# A = torch.randint(0, 10000, (1, 10000, 64))


# # mha = MultiHeadClusterAttention()
# # x = mha(x, A)

# Cluster_vit = ClusterViT(input_emb_size=1024, emb_size=512, n_classes=2)
# y = Cluster_vit(x, A)

# print ('done!')