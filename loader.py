import os
import os.path as osp
import time

import matplotlib.pyplot as plt

import os.path as osp
import os

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
if sys.version_info[0] == 2:
	import cPickle as pickle
else:
	import pickle

class Dataset_Gary(Dataset):

	def __init__(self, root, fold="train",
				 transform=None, target_transform=None):
		
		fold = fold.lower()

		self.train = False
		self.test = False
		self.val = False

		if fold == "train":
			self.train = True
		elif fold == "test":
			self.test = True
		elif fold == "val":
			self.val = True
		else:
			raise RuntimeError("Not train-val-test")


		self.root = os.path.expanduser(root)
		self.transform = transform
		self.target_transform = target_transform

		# now load the picked numpy arrays
		self.data = []
		if self.train:
			self.datalist_dir = os.path.join(self.root, 'train_list.txt')
		if self.val:
			self.datalist_dir = os.path.join(self.root, 'val_list.txt')
		if self.test:
			self.datalist_dir = os.path.join(self.root, 'test_list.txt')

		with open(self.datalist_dir, 'r') as f:
			for line in f:
				if line[0] == '#' or len(line.strip()) == 0:
					continue
				params = line.strip().split()
				self.data.append({
					'file_name' : params[0],
					'label' : params[1],})

	def __getitem__(self, index):
		label = self.data[index]['label']
		if label == 'cardboard':
			target = 0
		if label == 'glass':
			target = 1
		if label == 'metal':
			target = 2
		if label == 'paper':
			target = 3
		if label == 'plastic':
			target = 4
		if label == 'trash':
			target = 5
		img = plt.imread(osp.join(self.root, self.data[index]['label'], self.data[index]['file_name']))

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data)
		