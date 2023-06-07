import torch
import torch.nn as nn
import torchvision




class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder_base =  torchvision.models.__dict__['resnet18'](pretrained=True)
        self.decoder_base =  torchvision.models.__dict__['resnet18'](pretrained=True)



