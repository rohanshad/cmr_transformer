'''
Pytorch Dataset classes for public release
'''

from pickletools import TAKEN_FROM_ARGUMENT4U
import torch
import argparse as ap
import pandas as pd
import pytorch_lightning as pl
import os
import h5py
import numpy as np
from transformers import AutoTokenizer
from torchvideotransforms import video_transforms, volume_transforms
import json
import glob
import random
pd.options.mode.chained_assignment = None

import nibabel as nib
import functools
from nibabel.testing import data_path
from pytorchvideo import transforms
import sys
from sklearn.model_selection import GroupShuffleSplit
       
'''
General torch.utils.data.Dataset class for whole project
Input master_csv must have the following columns: 
| filename | split | institution |

hdf5 files must be preprocessed / built to contain pre-specified views:
stanford_RF3da3244
├── RF3da3581.h5
├── RF3lv2173.h5
	├── 4CH 		{data: 4d array} {attr: fps, total images}
	├── SAX			{data: 4d array} {attr: fps, total images, slice frame index}
	├── 3CH			{data: 4d array} {attr: fps, total images}
'''

class MRI_torch_dataset(torch.utils.data.Dataset):
	'''
	Main torch dataloader for MRI ML projects
	
	Args:
		root (str):         	Parent directory for hdf5 data directory
		split (str):        	Train / Val / Test splits
		task (str):         	Specify training / finetuning task
		mean (float):       	Mean pixel value 
		sd (float):         	Standard deviation for pixel values
		frames (float):     	Total number of frames 
		period (float):     	Temporal subsampling rate 
		max_length (float): 	Maximum video length to take before trimming 
		report_datafile (str):	Name of report_datafile
		tokenizer (str):    	BERT tokenizer to use for text report 
		mri_views (str / list): Views to pull for training
	'''

	def __init__(self, root, mri_folder, master_csv, split, task, frames, period, max_length, tokenizer, mri_views, report_datafile, sentences, transform_map, percentage, debug, manual_seed, target, drop_samples):
		self.root = root
		self.mri_folder = mri_folder
		self.master_csv = master_csv
		self.split = split
		self.task = task
		self.frames = frames
		self.period = period
		self.max_length = max_length
		if tokenizer is not None:
			self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
		self.transforms = transform_map
		self.mri_views = mri_views
		self.report_datafile = report_datafile
		self.percentage = percentage
		self.debug = debug
		self.num_sentences = sentences
		self.manual_seed = manual_seed 
		self.target = target
		self.drop_samples = drop_samples

		# Load master_csv sheet
		master_sheet = pd.read_csv(os.path.join(self.root, self.master_csv))
		available_folders = os.listdir(os.path.join(self.root, self.mri_folder))

		# Keep only a percentage subset of data
		if self.percentage != 1.0:
			master_sheet = self.generate_percentage_subsample(master_sheet, self.percentage)

		# Set list for mri_views
		if type(self.mri_views) is not list:
			self.mri_views = [self.mri_views]

		# Figure out which datasource to use for this project
		if self.split == "external_test":
			# Filenames are tbd - will have to edit this section based on folder structure that I get 
			self.filenames = sorted(os.listdir(self.root))
		else:
			data = master_sheet[master_sheet['split'] == self.split]
			# Filenames are relative paths to accession number: [mrn-acc_num.h5]
			data['filepath'] = data['parent_folder']+'/'+data['accession']
			data['filenames'] = data['parent_folder']+'-'+data['accession']
			
			if self.task == "pretrain":
				data = data[data['view'].isin(self.mri_views)]
				# Print dataframe here for debugging 
				if self.debug == True:
					print(data[['parent_folder', 'accession', 'view', 'split']])

			elif self.task == "regresion" or self.task == "regression_mil":
				data = data.dropna(thresh=1)
				#Can have this bit editable through config files for lvedd / rvedd etc if someone wants
				self.target = data[target].tolist()
				if 'view' in data.columns:
					self.item_view = data['view'].tolist()
				else:
					self.item_view = None
			
			elif self.task == "classification_mil":
				data = data.dropna(thresh=1)
				#self.target = np.array(data[[i for i in TARGET_LIST]])
				self.target = np.array(data[target])
				if self.debug == True:
					print('------------------------------------')
					print(data)
					print(f'Targets: {self.target}')
				if 'view' in data.columns:
					self.item_view = data['view'].tolist()
				else:
					self.item_view = None


			self.header = data.columns.tolist()
			self.filenames = data['filenames'].tolist()

			
				

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, index):

		parent_folder, sep, accession = self.filenames[index].partition('-')
		accession_path = os.path.join(self.root, self.mri_folder, parent_folder, accession)

		if self.debug == True:
			print('Pulling video from:', self.filenames[index])
		
		# Read hdf5 file, concatenate arrays , and pull targets
		data = h5py.File(accession_path, 'r')

		# Gather targets for different tasks
		if self.task == "pretrain":
			'''
			Returns videos and paired tokenized text reports for contrastive pretraining to maximise mutual information
			Provides single video per iteration from a randomly selected slice (if available)
			'''

			print('Pre-training codebase coming soon...')

		else:

			if '_mil' in self.task:
				'''
				Returns list of videos for multi-instance learning / inference for classification problems
				Provides all views available for inference, picks every 2nd SAX slice
				Total videos per patient: (1x 4CH) + (1x 2CH) + (1x 3CH) + (4x SAX) = 7-8 videos
				'''
				view_list = []
				video_list = []
				for i in data.keys():
					if self.debug:
						print(f'Pulling video from {i}')
					if i in self.mri_views:
						if i == 'SAX':
							# Discards first and last slice of the SAX stack assuming they're trash
							slice_frames = data.get(i).attrs['slice_frames']
							if len(slice_frames) > 5:
								slice_frames = slice_frames[1:-1]
							for idx, s in enumerate(slice_frames):
								# Pull every second slice for performance, can tweak this later
								if self.drop_samples == True:
									if idx % 2 == 0:
										video, view = self.retrieve_mri_video(data, index, mri_view=i, input_slice=s)
										if video is not None:
											video_list.append(video)
											view_list.append(i)
								else:
									video, view = self.retrieve_mri_video(data, index, mri_view=i, input_slice=s)
									if video is not None:
										video_list.append(video)
										view_list.append(i)
						else:
							video, view = self.retrieve_mri_video(data, index, mri_view=i, input_slice='random')
							if video is not None:
								video_list.append(video)
								view_list.append(i)
			else:
				video_list, view_list = self.retrieve_mri_video(data, index, mri_view='random', input_slice='random')

			if self.task == "external_test":
				'''
				Potentially keep multi-instance video pull as a separate function
				'''
			
			elif self.task == "regression" or "regression_mil":
				'''
				Returns a list of videos for multi-instance learning / inference for regression problems.
				'''
				target = self.target[index]
				filenames = self.filenames[index]
				if len(video_list) < 1:
					video_list = None 

			elif self.task == "classification_mil":
				'''
				Pulls targets for multi-instance classification
				'''
				# target = [some sort of one-hot array]
				#target = filename, view_list, label
				target = self.target[index]
				filenames = self.filenames[index]
				if len(video_list) < 1:
					video_list = None 
			
			return video_list, view_list, target, filenames

		
		# Return video, target and filename later
		data.close()

	def retrieve_mri_video(self, data, index, mri_view, input_slice):
		'''
		Returns a video tensor [c, f, h, w] with our without transformations. 
		Represents single cine-slice from a spcific MRI view plane (self.item_view[index])
		'''

		'''
		Section below loads in slices of the arrays within the hdf5 files for 50-70% reduction in I/O overhead
		Essential for SAX or other views where there could be up to 300 frames in the video stack.
		'''
		if 'UniformTemporalSubsample' in str(self.transforms.transforms):
			temporal_subsample = True
		else:
			temporal_subsample = False

		if mri_view == 'random':
			view = self.item_view[index]
		else:
			view = mri_view

		if self.debug == True:
			print(f'Attempting to pull view: {view}')

		slice_frames = data[view].attrs['slice_frames']

		if len(slice_frames) < 1:		
			# Subsample every ith frame from array {float 32}
			array = np.array(data[view])

			if self.period > 1:
				array = np.array(data[view][:, 0:self.frames:self.period, :, :])
			
			if temporal_subsample == True:				
				array = np.array(data[view][:, 0::1, :, :])
			else:
				array = np.array(data[view][:, 0:self.frames:1, :, :])

			if self.debug == True:
				print(f'Size of array: {array.shape}')
				print(f'View: {view}')
				print(f'slice_frames: {slice_frames}')
		

		else:
			if view != 'SAX':
				mx = len(slice_frames)//2
				non_sax_frames = slice_frames[mx] - slice_frames[mx-1]
			
			if len(slice_frames) == 1:
				non_sax_frames = slice_frames[0]+1

			if len(slice_frames) > 5:
				if self.drop_samples == True:
					slice_frames = slice_frames[1:-1]
			
			slice_frames = slice_frames.tolist()

			# Subsample every ith frame from array {float 32} but start at random slice plane
			if input_slice == 'random':
				start = random.choice(slice_frames)
				if self.debug == True:
					print(slice_frames)
			else:
				start = input_slice

			if self.period > 1:		
				array = np.array(data[view][:, start:start+self.frames:self.period, :, :])
			
			if temporal_subsample == True:		
				if slice_frames.index(start) == len(slice_frames)-1:
					frames = abs(slice_frames[slice_frames.index(start)] - slice_frames[slice_frames.index(start)-1])
				else:
					frames = abs(slice_frames[slice_frames.index(start)+1] - slice_frames[slice_frames.index(start)])
				
				
				if frames == 0 and non_sax_frames:
					array = np.array(data[view][:, start:start+non_sax_frames:1, :, :])
				else:
					array = np.array(data[view][:, start:start+frames:1, :, :])

			else:
				array = np.array(data[view][:, start:start+self.frames:1, :, :])

			if self.debug == True:
				print(f'Size of array: {array.shape}')
				print(f'Number of frames calculated: {frames}')
				print(f'view: {view}')
				print(f'slice_frames: {slice_frames}')

		if self.transforms is not None:
			### SHOULD I TEST BY REMOVING THIS? ###
			if self.drop_samples == True:
				if array.shape[1] < self.frames:
					if self.debug == True:
						print(f'Quick_mafs: {array.shape[1]} - {self.frames}')
						print('Too few frames, skipping...')
					return None, None
				else:	
					video = self.transforms(torch.Tensor(array))
			else:
				video = self.transforms(torch.Tensor(array))
			
		else:
			video = array

		return video, view


	def generate_percentage_subsample(self, dataframe, percentage):
		'''
		Generates a % subsample of training data supplied for data efficiency experiments
		'''
		dataframe = dataframe.reset_index()
		keep_idx, discard_idx = next(GroupShuffleSplit(test_size=(1.0 - float(percentage)), n_splits=5, random_state=self.manual_seed).split(dataframe, groups=dataframe['parent_folder']))
		keep_df = dataframe.drop(discard_idx)
		return keep_df


class ACDC_torch_dataset(torch.utils.data.Dataset):
	'''
	Dataloader for training samples from ACDC MRI dataset. Available data: CINE SAX
	Bernard, O. et al. IEEE Trans. Med. Imaging 37, 2514–2525 (2018).
	
	File structure:
	Imaging files are stored as niftii 
	patient001
		├── patient001_4d.nii.gz 
		├── patient001_frame01_gt.nii.gz
	'''

	def __init__(self, root_dir, transform_map, debug):

		self.debug = debug 
		self.transforms = transform_map

		# Pull list of compatible files
		file_list = glob.glob(os.path.join(root_dir,'*','*'))
		filenames = []
		labels = []

		for f in file_list:
			if f[-9:] == '4d.nii.gz':
				filenames.append(f)

		for f in file_list:
			if f[-3:] == 'cfg':
				label = self.cfg_reader(f)['Group']
				labels.append(label)

		# Hardcode numeric values for each group here
		class_list = {'MINF': 1, 'RV': 2, 'NOR': 0, 'HCM': 3, 'DCM': 4}

		master_sheet = pd.DataFrame(list(zip(filenames, labels)), columns=['filenames', 'labels'])
		master_sheet['labels'] = master_sheet['labels'].map(class_list, na_action='ignore')

		self.filenames = master_sheet['filenames'].to_list()
		self.labels = master_sheet['labels'].to_list()

		print('------------------------------------')
		print('Total number of files:', len(self.filenames))
		print(master_sheet)
		print('------------------------------------')
			
	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, index):

		#Load nifti file
		df = nib.load(self.filenames[index])
		if self.debug == True:
			print(df.header)
			print('Array Shape:', np.array(df.dataobj).shape)

		# nifti arrays are images shaped in [h, w, slice, f]
		array = np.array(df.dataobj)
		slice_frames = range(2,array.shape[2]-2)
		#array = array[:,:,random.choice(range(array.shape[2])),:]
		array = array[:,:,3,:]

		# produce array shape [c, f, h, w]
		array = array.transpose(2, 0, 1)
		array = np.repeat(array[None,...],3,axis=0)
		video = torch.Tensor(array[:,0::1,:,:])
		
		# build target
		target = self.labels[index]
		target = torch.Tensor(np.eye(5)[target])


		# return transformed video, target and later filename 
		if self.transforms is not None:
			video = self.transforms(video)

		# Finally
		filename = os.path.basename(self.filenames[index])[:-7]
		return video, target, filename


	def cfg_reader(self, input_cfg):
		'''
		Quick .cfg file reader for acdc labels
		'''
		config = {}
		with open(input_cfg) as fp:
			for line in fp:
				key, val = line.strip().split(': ')
				config[key] = val

		return config		

if __name__ == "__main__":


	parser = ap.ArgumentParser(
		description="Dataset Debugger",
		epilog="Version 0.1; Created by Rohan Shad, MD"
		)

	parser.add_argument('-r', '--root_dir', metavar='', required=False, help='Full path to root directory', default='/scratch')
	parser.add_argument('-l', '--csv_list', metavar='', required=False, help='Process only files listed in csv_list.csv', default=None)
	parser.add_argument('-c', '--cpus', metavar='', type=int, default='4',help='number of cores to use in multiprocessing')
	parser.add_argument('-s', '--framesize', metavar='', type=int, default='224', help='framesize in pixels')
	parser.add_argument('--split', metavar='', default='train')
	parser.add_argument('--task', metavar='', default='pretrain')
	parser.add_argument('--report_datafile', metavar='', default='ucsf_stanford_bert_base_cased_oct18.json')
	args = vars(parser.parse_args())
	print(args)

	root_dir = args['root_dir']
	csv_list = args['csv_list']
	cpus = args['cpus']
	framesize = args['framesize']
	split = args['split']
	task = args['task']
	report_datafile = args['report_datafile']

	print("\nBegin DataLoader Testing ")

	# 0. miscellaneous prep
	torch.manual_seed(0)
	np.random.seed(0)


	# 1. create Dataset and DataLoader object
	print('------------------------------------')
	print("\nCreating Dataset and DataLoader.... ")

	# Apply video transformations ##
	tmp_transforms = transforms.create_video_transform(mode='train', 
			num_samples=None,
			max_size=380,
			min_size=framesize+40,
			crop_size=framesize,
			aug_type='default',
			convert_to_float=False,
			)
	div255 = transforms.Div255()
	temporal_sample = transforms.UniformTemporalSubsample(16)

	# Compose transformations
	transform_map_train = transforms.transforms_factory.Compose([temporal_sample] + [tmp_transforms] + [div255]) 
	
	def multi_instance_collate(batch):
		video_list = []
		view_list = []
		target_list = []
		filenames_list = []

		# Each sublist contains info from a single example within a batch
		for _video_sublist, _view_sublist, _target, _filename in batch:
			video_list.append([torch.Tensor(item) for item in _video_sublist])
			view_list.append(_view_sublist)
			target_list.append(_target)
			filenames_list.append(_filename)

		# Return list of torch tensors (videos), list of str (view), torch.float64 to models
		return video_list, view_list, torch.tensor(target_list, dtype=torch.float64), filenames_list

		
	def custom_collate_replace(batch, dataset):
		'''
		Custom collate function to get rid of samples that yield empty tensors
		Skips examples in a batch returning unequal batches, and pulls another new sample to replace it
		https://stackoverflow.com/questions/57815001/pytorch-collate-fn-reject-sample-and-yield-another
		'''
		len_batch = len(batch)
		# Filter out all Nones
		batch = list(filter (lambda x:x is not None, batch))

		if len_batch > len(batch):
			db_len = len(dataset)
			diff = len_batch - len(batch)
			while diff != 0:
				a = dataset[np.random.randint(0, db_len)]
				if a is None:
					continue
				batch.append(a)
				diff -= 1
		
		return multi_instance_collate(batch)
		#return torch.utils.data.dataloader.default_collate(batch)
	
	train_ds = MRI_torch_dataset(root=root_dir, mri_folder='ukbiobank_allviews_Nov30_2021', master_csv=csv_list, split=split, task=task, frames=20, period=1, max_length=96, tokenizer='bert-base-cased', mri_views=['4CH','3CH', '2CH','SAX'], report_datafile=report_datafile, sentences=3, transform_map=transform_map_train, percentage=1.0, debug=True, manual_seed=12345)
	custom_collate = functools.partial(custom_collate_replace, dataset=train_ds)

	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=1, collate_fn=custom_collate)

	# 2. Iterate through training data twice
	for epoch in range(1):
		print("Epoch = " + str(epoch))
		for (batch_idx, batch) in enumerate(train_loader):
			print("\nBatch = " + str(batch_idx))
			

			if task == 'pretrain':
				video = batch[0]
				encoded_text = batch[1]
			
				print(batch)

			elif task == 'diagnosis':
				video = batch[0]
				target = batch[1]
				filename = batch[2]

				df = pd.DataFrame(np.array([filename, target.to_numpy()]).transpose())
				df.columns = ['filename','target']
				print(df)


			elif task == 'regression_mil':
				print(batch)
				video_list = batch[0]
				target = batch[1]
				print(f'Printing batch statistics...')
				print(f'Batch size: {len(batch)}')
				print(f'First list of videos are of size: {len(video_list[0])}')
				print(f'dtype of videos: {type(video_list[0][0])}')

			else:
				print('Task not specified')

		print('------------------------------------')

		print("\nEnd test ")

