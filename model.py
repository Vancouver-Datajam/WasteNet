import torch
import torch.nn as nn
import torchvision.models as models

import os.path as osp
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import sys
if sys.version_info[0] == 2:
		import cPickle as pickle
else:
		import pickle

from skimage.transform import resize


class PreTrainedResNet(nn.Module):
	def __init__(self, num_classes, feature_extracting):
		super(PreTrainedResNet, self).__init__()
		
		self.resnet18 = models.resnet18(pretrained=True)

		if feature_extracting:
			for param in self.resnet18.parameters():
					param.requires_grad = False
		
		num_feats = self.resnet18.fc.in_features
		
		self.resnet18.fc =  nn.Linear(num_feats,num_classes)

	def forward(self, x):
		x = self.resnet18.forward(x)
		return x