import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F


class KN_fully_connected(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, model, num_ftrs, num_classes):
		super(KN_fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_1 = x
		out_3 = self.fc_4(x)
		return  out_1

class KimiaNet(nn.Module):
    def __init__(self, weights_location, feature_extracting=True):
        super(KimiaNet, self).__init__()

        self.model = torchvision.models.densenet121(pretrained=True)

        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False


        self.model.features = nn.Sequential(self.model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.model_final = KN_fully_connected(self.model.features, self.model.classifier.in_features, 30)
        print (f'self.model.classifier.in_features: {self.model.classifier.in_features}')
        self.model_final = nn.DataParallel(self.model_final)

        self.model_final.load_state_dict(
            torch.load(weights_location))

    def forward(self, x):
        return self.model_final(x)


# torch.cuda.set_device(2)
path = '/projects/ovcare/classification/parisa/projects/pretrained/KimiaNetPyTorchWeights.pth'
km_net = KimiaNet(weights_location=path, feature_extracting=True).cuda()
x = torch.rand((512, 3, 512, 512)).cuda()
out = km_net(x)
print (x.shape, out.shape)
