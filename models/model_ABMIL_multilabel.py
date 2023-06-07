import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, L = 1024, D = 128, n_labels = 4, n_classes = 2):
        super(Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = n_labels

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, self.L),
        #     nn.ReLU(),
        # )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        # self.classifier = nn.Linear(self.L*self.K, n_labels)
        bag_classifiers = [nn.Linear(self.L, 1) for i in range(n_labels)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.feature_extractor_part1 = self.feature_extractor_part1.to(device)
        # self.feature_extractor_part2 = self.feature_extractor_part2.to(device)
        self.attention = self.attention.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, x, return_features = False):
        device = x.device
        x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.feature_extractor_part2(H)  # NxL
        H = x

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        logits = torch.empty(1, self.K).float().to(device)
        for c in range(self.K):
            logits[:, c] = self.classifiers[c](M[c])

        Y_prob = torch.sigmoid(logits)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        features_dict = {}
        if return_features:
            features_dict['features'] = x

        return logits, Y_prob, Y_hat

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self, L = 1024, D = 128, n_labels = 4, n_classes = 2):
        super(GatedAttention, self).__init__()
        self.L = L
        self.D = D
        self.K = n_labels

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, self.L),
        #     nn.ReLU(),
        # )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        # self.classifier = nn.Linear(self.L*self.K, n_labels)
        bag_classifiers = [nn.Linear(self.L, 1) for i in range(n_labels)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.feature_extractor_part1 = self.feature_extractor_part1.to(device)
        # self.feature_extractor_part2 = self.feature_extractor_part2.to(device)
        self.attention_V = self.attention_V.to(device)
        self.attention_U = self.attention_U.to(device)
        self.attention_weights = self.attention_weights.to(device)
        self.classifiers = self.classifiers.to(device)        

    def forward(self, x, return_features = False):
        device = x.device
        x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.feature_extractor_part2(H)  # NxL
        H = x

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        logits = torch.empty(1, self.K).float().to(device)
        for c in range(self.K):
            logits[:, c] = self.classifiers[c](M[c])
        Y_prob = torch.sigmoid(logits)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        
        features_dict = {}
        if return_features:
            features_dict['features'] = x

        return logits, Y_prob, Y_hat

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A