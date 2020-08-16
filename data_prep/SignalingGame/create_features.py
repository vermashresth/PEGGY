import sys
#TODO: needs better way, in case they are severaldata_utils in pythonpath
from imagenet_data import produce_vgg_features
import random
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
import argparse

#TODO: can we know if the images are oriented the same?
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
	'--root', default=None, help='data root folder')
	parser.add_argument(
	'--out', default=None, help='data out folder')
	parser.add_argument(
	'--multi_layer', default=0, help='use multi layer message')
	opt = parser.parse_args()
	cuda = True
	random.seed(0)
	torch.manual_seed(0)
	np.random.seed(0)
	if cuda:
		torch.cuda.manual_seed_all(0)
		cudnn.benchmark = True
	produce_vgg_features(sftmax=0, multi_layer=opt.multi_layer, data=opt.root, save=opt.out, partition='test/')
	# produce_vgg_features(sftmax=1, data=opt.root, save=opt.out, partition='test/')
	produce_vgg_features(sftmax=0, multi_layer=opt.multi_layer, data=opt.root, save=opt.out, partition='train/')
	# produce_vgg_features(sftmax=1, data=opt.root, save=opt.out, partition='train/')
