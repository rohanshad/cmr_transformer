'''
Lightning Data Modules for public release
'''

import torch
import pandas as pd
import pytorch_lightning as pl
import os
import h5py
import numpy as np
from transformers import AutoTokenizer
from data.torch_datasets import MRI_torch_dataset, ACDC_torch_dataset
from pytorchvideo import transforms
import torchvision
import subprocess
import functools


class MRI_Data_Module(pl.LightningDataModule):
	'''
	MRI Data Module
	Common data module for variety of tasks (diagnosis, regression)
	Purpose of this module is to simply provide clean train / val / test sets for lightning modules
	References torch_datasets.py

	Args:
		data_dir: Root folder for datasets
		mri_folder: Folder within the root folder containing processed hdf5 MRI data
		master_csv: Main split_allocation.py csvfile 
		report_datafile: File containing report data as jsonl
		period: Variable to set the number of frames to skip (Period = 1 uses all frames)
		max_length: max input length of the video pulled
		tokenizer: Tokenizer to use (Might just remove this since data is pre-tokenized)
		task: Set the datamodule to pull samples for either contrastive pretraininig, diagnosis, or regression
		frames: Number of input frames to use 
		batch_size: Sets mini batchsize for training and validation
		mean: Mean pixel value for video data 
		sd: Standard deviation of pixel value for video data
		rotation_angle: Augmentation parameter for maximal rotation anlge
		shearing_angle: Augmentation parameter for shearing 
		mri_views: Sets the MRI views to use, choice between 4CH, 3CH, 2CH, or SAX
		crop_size: Input framesize to use 
		debug: Sets the dataloaders into debug mode 
		sentences: Number of input sentences to process via BERT for contrastive training 
		cpus: Number of worker CPUs to use

	'''

	def __init__(self, data_dir: str='None',
		mri_folder: str='None', master_csv: str='master_sheet_here.csv', report_datafile: str='None.json', period: int=1, max_length: int=96,
		tokenizer: str='None', task: str='classification_mil', frames: int=16, batch_size: int=16, mean: float=0.0, sd: float=1.0, rotation_angle: float=20.0,
		shearing_angle: float=5.0, mri_views: list=['4CH', '2CH', 'SAX', '3CH'], crop_size: int=224, debug: bool=False, augmentation_scheme: str='augmix', sentences: int=5, cpus: int=12, subsample_frames: bool=True, 
		percentage: float=1.0, manual_seed: int=42, cache: bool=False, target: str='Hypertrophic.Cardiomyopathy', drop_samples: bool=True): 
		super().__init__()

		self.data_dir = data_dir
		self.period = period
		self.max_length = max_length
		self.tokenizer = tokenizer
		self.task = task
		self.frames = frames
		self.batch_size = batch_size
		self.mean = mean
		self.sd = sd
		self.rotation_angle = rotation_angle
		self.shearing_angle = shearing_angle
		self.crop_size = crop_size
		self.num_workers = cpus
		self.mri_views = mri_views
		self.master_csv = master_csv
		self.report_datafile = report_datafile
		self.sentences = sentences
		self.mri_folder = mri_folder
		self.num_sentences = sentences
		self.debug = debug
		self.aug_settings = augmentation_scheme
		self.subsample_frames = subsample_frames
		self.percentage = percentage
		self.manual_seed = manual_seed
		self.cache_data = cache 
		self.target = target
		self.drop_samples = drop_samples

		### Calculate class imbalance ratio for classification tasks
		if self.task == "classification_mil":
			df = pd.read_csv(os.path.join(self.data_dir, self.master_csv))
			try:
				self.pos_weight = calculate_class_imbalance(df, self.target)
			except:
				self.pos_weight == 1.0

		### Video Transformations ###

		train_transforms = transforms.create_video_transform(mode='train', 
			num_samples=None,
			max_size=380,
			min_size=self.crop_size+40,
			crop_size=self.crop_size,
			aug_type=self.aug_settings,
			convert_to_float=False,
			)
		
		val_transforms = transforms.create_video_transform(mode='val', 
			num_samples=None,
			max_size=380,
			min_size=self.crop_size+40,
			crop_size=self.crop_size,
			aug_type=self.aug_settings,
			convert_to_float=False,
			)

		# String together the video transforms
		div255 = transforms.Div255()
		temporal_sample = transforms.UniformTemporalSubsample(self.frames)
		
		# Temporal sample BEFORE augmix = 2x speedup in performance
		# Is there a better way to do this on GPU freeing up the CPUs? 
		if self.subsample_frames == True:
			self.transform_map_train = transforms.transforms_factory.Compose([temporal_sample] + [train_transforms] + [div255]) 
			self.transform_map_val = transforms.transforms_factory.Compose([temporal_sample] + [val_transforms] + [div255]) 
		
		else:
			self.transform_map_train = transforms.transforms_factory.Compose([train_transforms] + [div255]) 
			self.transform_map_val = transforms.transforms_factory.Compose([val_transforms] + [div255]) 


		'''
		RANDAUG: https://arxiv.org/abs/1909.13719
		AUGMIX: https://arxiv.org/abs/1912.02781
		MIXUP: https://arxiv.org/abs/1710.09412
		'''

	def prepare_data(self):
		'''
		Cache data into high performance SSD if self.cache_data == TRUE
		'''
		if self.cache_data:
			print('Preparing Local SSD...')
			cache_dir = '/cache/cmr_data_cache'
			rclone_load = 'module load system rclone'

			### Copy data and accessory csv files ###
			rclone_sync_data = f'rclone sync  {os.path.join(self.data_dir, self.mri_folder)} {os.path.join(cache_dir, self.mri_folder)} --transfers {self.num_workers*4} -P --stats-one-line'
			rclone_sync_reports = f'rclone sync  {os.path.join(self.data_dir, self.report_datafile)} {cache_dir} -P --stats-one-line'
			rclone_sync_csv = f'rclone sync  {os.path.join(self.data_dir, self.master_csv)} {cache_dir} -P --stats-one-line'

			
			p = subprocess.Popen('{cmd1};{cmd2};{cmd3};{cmd4}'.format(cmd1=rclone_load, cmd2=rclone_sync_data, cmd3=rclone_sync_reports, cmd4=rclone_sync_csv), shell=True)
			streamdata = p.communicate()[1]

			self.data_dir = cache_dir

	def setup(self, stage=None):
		if stage == 'fit' or stage is None:
			self.train_set = MRI_torch_dataset(
				root=self.data_dir,
				mri_folder=self.mri_folder,
				master_csv=self.master_csv,
				split='train', 
				task=self.task, 
				mri_views=self.mri_views,
				frames=self.frames, 
				period=self.period, 
				max_length=self.max_length, 
				tokenizer=self.tokenizer,
				report_datafile=self.report_datafile,
				sentences=self.num_sentences,
				transform_map=self.transform_map_train,
				percentage=self.percentage,
				debug=self.debug,
				manual_seed=self.manual_seed,
				target=self.target,
				drop_samples=self.drop_samples
				)

		#if stage == 'validate' or stage is None:
			self.val_set = MRI_torch_dataset(
				root=self.data_dir,
				mri_folder=self.mri_folder,
				master_csv=self.master_csv,
				split='val', 
				task=self.task, 
				mri_views=self.mri_views,
				frames=self.frames, 
				period=self.period, 
				max_length=self.max_length, 
				tokenizer=self.tokenizer,
				report_datafile=self.report_datafile,
				sentences=self.num_sentences,
				transform_map=self.transform_map_val,
				percentage=1.0,
				debug=self.debug,
				manual_seed=self.manual_seed,
				target=self.target,
				drop_samples=self.drop_samples
				)

		elif stage =='test' or stage is None:
			self.test_set = MRI_torch_dataset(
				root=self.data_dir,
				mri_folder=self.mri_folder,
				master_csv=self.master_csv,
				split='test', 
				task=self.task, 
				mri_views=self.mri_views,
				frames=self.frames, 
				period=self.period, 
				max_length=self.max_length, 
				tokenizer=self.tokenizer,
				report_datafile=self.report_datafile,
				sentences=self.num_sentences,
				transform_map=self.transform_map_val,
				percentage=1.0,
				debug=self.debug,
				manual_seed=self.manual_seed,
				target=self.target,
				drop_samples=self.drop_samples
				)


	def train_dataloader(self):
		custom_collate = functools.partial(custom_collate_replace, dataset=self.train_set, task=self.task)
		train_loader = torch.utils.data.DataLoader(
			dataset=self.train_set,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			collate_fn=custom_collate,
			pin_memory=False,
			drop_last=True
			)

		print('Total training samples:', len(train_loader)*self.batch_size)
		return train_loader

	def val_dataloader(self):
		custom_collate = functools.partial(custom_collate_replace, dataset=self.val_set, task=self.task)
		val_loader = torch.utils.data.DataLoader(
			dataset=self.val_set,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			collate_fn=custom_collate,
			pin_memory=False,
			drop_last=True
			)

		print('Total val samples:', len(val_loader)*self.batch_size)
		return val_loader

	def test_dataloader(self):
		custom_collate = functools.partial(custom_collate_replace, dataset=self.test_set, task=self.task)
		test_loader = torch.utils.data.DataLoader(
			dataset=self.test_set,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			collate_fn=custom_collate,
			pin_memory=False,
			drop_last=True
			)

		print('Total test samples:', len(test_loader)*self.batch_size)
		return test_loader


class ACDC_Data_Module(pl.LightningDataModule):
	'''
	ACDC Dataset Module
	Supplementary datamodule for ACDC dataset sourced from: Bernard, O. et al. IEEE Trans. Med. Imaging 37, 2514–2525 (2018).
	Purpose of this module is to pull labelled cine SAX sequences for sanity checking clustering before full dataset is ready
	References torch_datasets.py
	'''

	def __init__(self, data_dir: str='/scratch/acdc_labelled_sax', frames: int=16, period: int=1, max_length: int=96, 
		task: str='acdc', batch_size: int=16, percentage: float=1.0, mean: float=0.0, sd: float=1.0,
		crop_size: int=224, debug: bool=False, subsample_frames: bool=True, cpus: int=8):
		super().__init__()

		self.root_dir = data_dir
		self.frames = frames
		self.crop_size = crop_size
		self.batch_size = batch_size
		self.num_workers = cpus
		self.subsample_frames = subsample_frames

		# Build dataset transformations
		acdc_transforms = transforms.create_video_transform(mode='val', 
			num_samples=None,
			max_size=380,
			min_size=self.crop_size+40,
			crop_size=self.crop_size,
			aug_type='default',
			convert_to_float=False,
			)

		# String together the video transforms
		div255 = transforms.Div255()
		if self.subsample_frames == True:
			temporal_sample = transforms.UniformTemporalSubsample(self.frames)
			self.transform_map = transforms.transforms_factory.Compose([acdc_transforms] + [div255] + [temporal_sample]) 
		else:
			self.transform_map = transforms.transforms_factory.Compose([acdc_transforms] + [div255]) 

	def setup(self, stage=None):
		if stage == 'validate' or stage is None:
			self.acdc_data = ACDC_torch_dataset(
				root_dir = self.root_dir,
				transform_map = self.transform_map,
				debug = False
				)
		if stage == 'predict' or stage is None:
			self.acdc_data = ACDC_torch_dataset(
				root_dir = self.root_dir,
				transform_map = self.transform_map,
				debug = False
				)

	def val_dataloader(self):
		val_loader = torch.utils.data.DataLoader(
			dataset=self.acdc_data,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			#collate_fn=custom_collate,
			pin_memory=True
			)

		print('Total val samples:', len(val_loader)*self.batch_size)
		return val_loader

	def predict_dataloader(self):
		pred_loader = torch.utils.data.DataLoader(
			dataset=self.acdc_data,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			#collate_fn=custom_collate,
			pin_memory=True
		)

		print('Total val samples:', len(pred_loader)*self.batch_size)
		return pred_loader

def calculate_class_imbalance(df, target):
	'''
	Calculates class imbalance from a supplied dataframe
	'''
	pos_weight = len(df[df['split']=='train'])/df[df['split']=='train'][target].sum()
	return pos_weight


def multi_instance_collate(batch):
	video_list = []
	view_list = []
	target_list = []
	filename_list = []

	# Each sublist contains info from a single example within a batch
	for _video_sublist, _view_sublist, _target, _filename in batch:
		try:
			video_list.append(torch.stack([torch.Tensor(item) for item in _video_sublist]))
			view_list.append(_view_sublist)
			target_list.append(_target)
			filename_list.append(_filename)
		except Exception as ex:
			print(f'Skipping {_filename}: enable print statements for more')
			# print(_video_sublist)
			# print(_view_sublist)
			# print(_target)
			# print(_filename)
			
	# Return list of torch tensors (videos), list of str (view), torch.float64 to models
	# This specific line is apparently very slow but I can't figure out how to fix it yet and I also don't care at this point
	
	return torch.cat(video_list), view_list, torch.tensor(target_list, dtype=torch.float32), filename_list
		
def custom_collate_replace(batch, dataset, task):
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
	
	if task == 'pretrain':
		return torch.utils.data.dataloader.default_collate(batch)

	elif '_mil' in task:
		try:
			return multi_instance_collate(batch)
		
		except Exception as ex:
			print(f'Data skipped by q/c steps, retrying another...')
			batch = []
			retries = 0
			retry_limit = 10

			while retries < retry_limit:
				try:
					db_len = len(dataset)
					a = dataset[np.random.randint(0, db_len)]
					if a is None:
						continue
					batch.append(a)
					return multi_instance_collate(batch)
					break
				except Exception as ex:
					print(f'ERROR FROM CUSTOM COLLATE: {ex}')
					retries += 1
					if retries == retry_limit:
						raise ex

	else:
		return torch.utils.data.dataloader.default_collate(batch)


