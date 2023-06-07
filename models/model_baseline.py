from turtle import forward
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from models.resnet_custom import resnet50_baseline


class FC(nn.Module):
    def __init__(self, model_size = "medium", dropout = False, n_classes = 2):
        super(FC, self).__init__()
        self.size_dict = {"small": 512, "medium": 1024, "big": 2048}
        self.fc = nn.Linear(self.size_dict[model_size], n_classes)
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc.to(device)
    
    def forward(self, x, return_features = False):
        logits = self.fc(x)
        y_probs = F.softmax(logits, dim = 1)
        y_hats = torch.argmax(y_probs, dim=1)


        features_dict = {}
        if return_features:
            features_dict['features'] = x
        
        return logits, y_probs, y_hats, features_dict


class Baseline(nn.Module):
    def __init__(self, n_classes = 2, model_size = "medium", feature_extracting = False):
        super(Baseline, self).__init__()
        self.size_dict = {"small": 512, "medium": 1024, "big": 2048}
        baseline = []
        model = resnet50_baseline(pretrained=True)
        dim = self.size_dict[model_size]

        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

        linear = nn.Linear(dim, n_classes)

        baseline = [model, linear]
        self.classifier = nn.Sequential(*baseline)
        initialize_weights(self)


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

    def forward(self, x, return_features = False):
        logits = self.classifier(x)
        y_probs = F.softmax(logits, dim = 1)
        y_hats = torch.argmax(y_probs, dim=1)


        features_dict = {}
        if return_features:
            features_dict['features'] = x
        
        return logits, y_probs, y_hats, features_dict


class Kather(nn.Module):
    def __init__(self, arch = 'shufflenet_v2_x1_0', n_classes = 2, feature_extracting = False):
        super(Kather, self).__init__()
        model = torchvision.models.__dict__[arch](pretrained=True)
        dim = model.fc.weight.shape[1]
        if arch == 'inception_v3':
            model.aux_logits = False
        model.fc = nn.Identity()

        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

        self.n_classes = n_classes
        linear = nn.Linear(dim, n_classes)
        baseline = [model, linear]
        self.classifier = nn.Sequential(*baseline)
        initialize_weights(self)


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

    def forward(self, x, return_features = False):
        logits = self.classifier(x)
        y_probs = F.softmax(logits, dim = 1)
        y_hats = torch.argmax(y_probs, dim=1)


        features_dict = {}
        if return_features:
            features_dict['features'] = x
        
        return logits, y_probs, y_hats, features_dict








