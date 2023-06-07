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

class MultiHeadLocalAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()
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
        
        # free_gpu_cache()
        
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        b, n, embed_size = x.shape
        b, n, N = A.shape
        scaling = self.emb_size ** (1/2)

        l = []
        Att = []

        for i in range(n):
            local_keys = torch.index_select(keys, dim=2, index=torch.squeeze(A[:,i,:])) #torch.Size([1, 8, 16, 64])
            local_values = torch.index_select(values, dim=2, index=torch.squeeze(A[:,i,:])) #torch.Size([1, 8, 16, 64])
            q = queries[:,:,i:i+1, :] #torch.Size([1, 8, 64])
            energy = torch.einsum('bhqd, bhkd -> bhqk', q, local_keys) # batch, num_heads, query_len, key_len
            att = F.softmax(energy, dim=-1) / scaling
            att = self.att_drop(att)
            patch_att = torch.sum(att, dim = (1, 2, 3))
            Att.append(patch_att)
            out = torch.einsum('bhal, bhlv -> bhav ', att, local_values)
            out = rearrange(out, "b h n d -> b n (h d)")
            l.append(out)
    
        # print(torch.cuda.memory_allocated() / 1024**3)
        l = torch.cat(l, dim = 1)
        Att = torch.stack(Att)

        # # sum up over the last axis
        # energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        # if mask is not None:
        #     fill_value = torch.finfo(torch.float32).min
        #     energy.mask_fill(~mask, fill_value)
            
        # scaling = self.emb_size ** (1/2)
        # att = F.softmax(energy, dim=-1) / scaling
        # att = self.att_drop(att)
        # # sum up over the third axis
        # out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # out = rearrange(out, "b h n d -> b n (h d)")
        # out = self.projection(out)
        return l, Att


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

class LocalTransformerEncoder(nn.Module):
    def __init__(self,
                 input_emb_size: int = 1024,
                 emb_size: int = 512,
                 drop_p: float = 0.,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super(LocalTransformerEncoder, self).__init__()

        # super().__init__(
        #     FeedForwardBlock(emb_size=input_emb_size, target_emb_size=emb_size, drop_p=forward_drop_p),
        #     ResidualAdd(nn.Sequential(
        #         MultiHeadLocalAttention(emb_size, **kwargs),
        #         nn.Dropout(drop_p)
        #     )),
        #     nn.LayerNorm(emb_size),
        #     ResidualAdd(nn.Sequential(
        #         MultiHeadLocalAttention(emb_size, **kwargs),
        #         nn.Dropout(drop_p)
        #     )),
        #     nn.LayerNorm(emb_size)
        #     )


        self.fc = FeedForwardBlock(emb_size=input_emb_size, target_emb_size=emb_size, drop_p=forward_drop_p)
        self.mhla_1 = MultiHeadLocalAttention(emb_size, **kwargs)
        self.ln_1 = nn.LayerNorm(emb_size)

        self.mhla_2 = MultiHeadLocalAttention(emb_size, **kwargs)
        self.ln_2 = nn.LayerNorm(emb_size)

        self.dropout = drop_p

    def forward(self, x, A, **kwargs):
        embs = self.fc(x)
        block_1, _ = self.mhla_1(embs, A[:, :, :16], **kwargs)
        block_1 = F.dropout(block_1, self.dropout)
        # Att_1 = F.dropout(Att_1, self.dropout)

        block_1 += embs
        block_1 = self.ln_1(block_1)

        block_2, Att_2 = self.mhla_2(block_1, A[:, :, :64], **kwargs)
        block_2 = F.dropout(block_2, self.dropout)
        Att_2 = F.dropout(Att_2, self.dropout)

        block_2 += block_1
        block_2 = self.ln_2(block_2)

        return block_2, Att_2




class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 512, n_classes: int = 2, n_labels: int = 4):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size))



class LocalViT(nn.Module):
    def __init__(self,     
                L: int = 1024,
                D: int = 512,
                n_classes: int = 2,
                n_labels: int = 4,
                **kwargs):
        super(LocalViT, self).__init__()
        self.encoder = LocalTransformerEncoder(input_emb_size=L, emb_size=D, **kwargs)
        self.classifier = ClassificationHead(D, n_classes, n_labels)
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.encoder.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, x, A):
        embs, Atts = self.encoder(x, A)
        return Atts, torch.squeeze(embs)
    
    def compute_agg(self, A, embs):
        embs = embs[None, :, :]
        feats = self.classifier(embs)
        return feats





# x = torch.rand(1, 10000, 1024)
# A = torch.randint(0, 10000, (1, 10000, 64))


# # mha = MultiHeadLocalAttention()
# # x = mha(x, A)

# local_vit = LocalViT(input_emb_size=1024, emb_size=512, n_classes=2)
# y = local_vit(x, A)

# print ('done!')