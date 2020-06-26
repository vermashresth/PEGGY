import argparse
import os
import numpy as np
import shutil

def main(opt):
	root_dir = opt.root
	out_dir = opt.out
	concepts = os.listdir(root_dir)
	for c in concepts:
	  if not os.path.exists(out_dir +'/train/' + c):
	    os.makedirs(out_dir +'/train/' + c)
	  if not os.path.exists(out_dir +'/test/' + c):
	    os.makedirs(out_dir +'/test/' + c)

	  print(c)
	  currentCls = c
	  src = root_dir + '/' +currentCls # Folder to copy images from

	  allFileNames = os.listdir(src)
	  np.random.shuffle(allFileNames)
	  n_files = len(allFileNames)
	  train_FileNames = allFileNames[:int(0.8*n_files)]
	  test_FileNames = allFileNames[int(0.8*n_files):]


	  train_FileNames = [src+'/'+ name for name in train_FileNames]
	  test_FileNames = [src+'/' + name for name in test_FileNames]

	  # Copy-pasting images
	  for name in train_FileNames:
	      if name.endswith('.png') or name.endswith('.jpg'):
		      shutil.copy(name, out_dir +'/train/' + c)

	  for name in test_FileNames:
	      if name.endswith('.png') or name.endswith('.jpg'):
		      shutil.copy(name, out_dir +'/test/' + c)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
	'--root', default=None, help='data root folder')
	parser.add_argument(
	'--out', default=None, help='data out folder')
	opt = parser.parse_args()
	main(opt)


