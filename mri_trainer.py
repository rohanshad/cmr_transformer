'''
mri_trainer.py lightning module for public release
'''

import torch
import os
from torch import nn as nn
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.cli import instantiate_class
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar 
import torchmetrics
from transformers import AutoModel
import model_factory
from data.data_modules import MRI_Data_Module, ACDC_Data_Module


import wandb
from wandb.sdk.data_types._dtypes import AnyType
import numpy as np

import seaborn as sns
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from jsonargparse import lazy_instance
from sklearn.manifold import TSNE
from umap import UMAP

from pytorch_lightning.loggers import WandbLogger
from pyaml_env import BaseConfig, parse_config
import platform
import time

# Read local_config.yaml for local variables 
device = platform.uname().node.replace('-','_')
cfg = BaseConfig(parse_config('local_config.yaml'))
if 'sh' in device:
	device = 'sherlock'
elif '211' in device:
	device = 'cubic'
ATTN_DIR = getattr(cfg, device).attn_dir
WANDB_CACHE_DIR = getattr(cfg, device).tmp_dir
plt.style.use('classic')
os.makedirs(ATTN_DIR, exist_ok = True)


class Cardiac_MRI_Net(LightningModule):
	'''
	Core module
	Downstream evaluation and finetuning for NCE-pretrained models

	Args:
		video_model: Video based deep learning architechture to use 
		pretrained: Initialize architecture with contrastive pre-trained weights
		video_model_depth: Number of layers for architecture family 
		checkpoint_path: full path of checkpoint to initialize with
		multi_view_strategy: variable to dictate strategy for combining multiple views
		batch_size: linked with data.module batch_size variable
		proj_dim: MLP projection head dimensional size
		proj_layers: Projection layrers to use for MLP
		dropout: Set dropout for projection layers
		lr: Learning rate for video model
		mode: Linear evaluation, finetune, or transfer learn
		output_classes: Output categories for classification performance metrics
		wd: Weight decay value for video model
		learning_schedule: Learning rate schedule to use
		steplr_decaysteps: Number of epochs after which decay learning rate
		steplr_decaygamma: LR decay factor

	'''


	def __init__(self, video_model: str='mvit',  pretrained: bool=True, video_model_depth: int=50, checkpoint_path: str='path_to_checkpint.ckpt', 
		multi_view_strategy: str='single_model', lr: float=0.0001, wd: float=0.0001, batch_size: int=24, proj_dim: int=512, proj_layers: int=1,dropout: float=0.1, mode: str='linear', output_classes: int=1,
		optimizer_init: dict={None}, lr_scheduler_init: dict={None}, frames: int=16, multi_instance: bool=False, task: str='classification', data_percentage: float=1.0, print_attention_maps: bool=False, layernorm: bool=False, pos_weight: float=1.0, target: str='Something'):
		super().__init__()
		self.save_hyperparameters()

		self.multi_instance = multi_instance
		self.batch_size = batch_size
		self.data_percentage = data_percentage
		self.task = task
		self.print_attention_maps = print_attention_maps

		# AdamW and StepLR setup (deals with lr and wd now)
		self.optimizer_init = optimizer_init
		self.lr_scheduler_init = lr_scheduler_init

		# Load contrastive pre-trained models based on how multiple views are processed:
		if multi_view_strategy == 'single_model':
			self.model = model_factory.Cardiac_MRI_Encoder(video_model, video_model_depth, proj_dim, proj_layers, 5, frames, mode, True)
			if pretrained == True:
				print('------------------------------------')
				print('Loading contrastive pre-trained weights...')
				self.model = model_factory.load_contrastive_pretrained_weights(self.model, checkpoint_path)

		# Multi-instance learning
		if self.multi_instance == True:
			# Make sure trainer uses batchsize of 1 for multi-instance learning 
			assert batch_size == 1
			self.classifier_head = model_factory.Multiview_Attention(n_classes=output_classes, task=self.task, normalize=layernorm)
	
		# Debug prints
		print(f'Task set to: {self.task}')
		print(f'Mode set to: {mode}')

		# Loss function for multi-task learning:
		print(f'POS_WEIGHT CALCULATED: {pos_weight}')
		self.classif_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(np.array(pos_weight)))
		self.regress_loss = nn.HuberLoss()

		# Metrics
		self.accuracy = torchmetrics.Accuracy(task='binary')
		self.f1 = torchmetrics.F1Score(task='binary')
		self.auroc = torchmetrics.AUROC(task='binary')
		self.val_f1 = torchmetrics.F1Score(task='binary')
		self.val_auroc = torchmetrics.AUROC(task='binary')
		self.test_auroc = torchmetrics.AUROC(task='binary')
		self.mae = torchmetrics.MeanAbsoluteError()
		self.mse = torchmetrics.MeanSquaredError()

		# Init outputs for on_epoch_end funnctions
		self.training_step_outputs = []
		self.validation_step_outputs = []
		self.test_step_outputs = []


	def forward(self, video_batch, return_mil_attn=False, generate_attnmaps=False):
		
		#if self.multi_instance == True:
		if generate_attnmaps == True:
			video_emb, attn_list = self.model(video_batch, generate_attnmaps)
			'''
			Current visualization module is good to take a single view froma a single hdf5 file
			Not yet designed to generate videos from a batch of videos from same hdf5 file source
			
			### TODO ###
			Present workflow:
			single video [3, 16, 224, 224] >> attention from model stored in hdf5 

			Future workflow: 
			batch of videos [3, 16, 224, 224] >> save attentions from models in hdf5 in batches
			export to [3, 8, x, y] separately once epoch has ended / post-processing step
			'''

		else:
			video_emb = self.model(video_batch)
		
		logits, probs, Y_hat, attn_raw, results_dict = self.classifier_head(video_emb, self.task)

		if return_mil_attn == True:
			return logits, probs, Y_hat, video_emb, attn_raw
		elif generate_attnmaps == True:
			return logits, probs, Y_hat, video_emb, attn_list
		else:
			return logits, probs, Y_hat, video_emb

	def predict_step(self, batch, batch_idx):
		'''
		Return attentions by default as a list of torch.Tensors
		'''
		video, labels, filenames = batch
		video_emb, attn_list = self.model(video, generate_attnmaps=True)		
		
		self._hdf5_writer(video, attn_list, filenames)


	def training_step(self, batch, batch_idx):
		inputs, views, target, _ = batch		
		if 'regression_mil' in self.task:
			labels = torch.unsqueeze(target, dim=0).float()

		elif 'classification_mil' in self.task:
			#labels = torch.squeeze(target, dim=0).int()
			labels = torch.unsqueeze(target, dim=0).int()
			# print(f'LABELS: {labels}')
			# print(f'LABELS SHAPE: {labels.shape}')
			# print(f'INPUTS: {inputs.shape}')
		else:
			labels = torch.unsqueeze(target, dim=1).float()	

		logits, probs, Y_hat, embeddings = self.forward(inputs, return_mil_attn=False)
		self.log("step", self.global_step, batch_size=self.batch_size)

		# print(inputs.shape)
		if self.task == 'classification' or 'classification_mil':
			loss = self.classif_loss(logits, labels.float())
			accuracy = self.accuracy(logits, labels)

			# Update AUC & F1
			self.auroc.update(logits, labels)
			self.f1.update(Y_hat, labels)

			# Log non-modular metrics
			self.log('acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

		elif self.task == 'regression' or self.task == 'regression_mil':
			# print(result)
			loss = self.regress_loss(Y_hat, labels)
			mae = self.mae(Y_hat, labels)
			mse = self.mse(Y_hat, labels)

			# Log all metrics
			self.log('mae', mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
			self.log('mse', mse, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

		# Log losses
		self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

		self.training_step_outputs.append(loss)
		return loss

	def on_training_epoch_end(self):
		'''
		Empty function to make torchmetrics behave well
		https://github.com/Lightning-AI/torchmetrics/pull/907/files?short_path=95b021e#diff-95b021efc2d4ea76bba0062affce2cae7f4712f81982c64329432b6d15616238
		'''
		outputs = self.training_step_outputs
		self.log('auc_epoch', self.auroc.compute(), prog_bar=True, sync_dist=True)
		self.log('f1_epoch', self.f1.compute(), prog_bar=True, sync_dist=True)
		self.auroc.reset()
		self.f1.reset()

		self.training_step_outputs.clear()
		outputs.clear()

	def validation_step(self, batch, batch_idx):
		inputs, views, target, filenames = batch
		if 'regression_mil' in self.task:
			labels = torch.unsqueeze(target, dim=0).float()	

		elif 'classification_mil' in self.task:
			#labels = torch.squeeze(target, dim=0).int()
			labels = torch.unsqueeze(target, dim=0).int()
			# print(f'LABELS: {labels}')
			# print(f'LABELS SHAPE: {labels.shape}')
			# print(f'INPUTS: {inputs.shape}')

		else:
			labels = torch.unsqueeze(target, dim=1).float()	

		if self.task == 'acdc': 
			video, target, filenames = batch
			_, _, _, video_emb, attn_list = self.forward(video, generate_attnmaps=True)
			self._hdf5_writer(video, attn_list, None, filenames)		
			print('Exported attention maps from', len(attn_list), 'layers')

			self.validation_step_outputs.append([video_emb, target])
			return video_emb, target

		else:
			# Uses the multi-instance attention classifier	
			logits, probs, Y_hat, embeddings, attn_list = self.forward(inputs, return_mil_attn=True)
			
			# print(f'views: {views}')
			# print(f'raw_attn: {attn_list.cpu().detach().numpy()}')
			# print(f'softmax_attn: {nn.functional.softmax(attn_list, dim=1).cpu().detach().numpy()}')


			if self.task == 'classification' or self.task == 'classification_mil':
				val_loss = self.classif_loss(logits, labels.float())
				val_accuracy = self.accuracy(logits, labels)

				# AUC update; compute later and log later
				self.val_auroc.update(logits, labels)
				self.val_f1.update(Y_hat, labels)
				metrics = val_loss, val_accuracy 

				# Log non-modular metrics and losses
				self.log('val_acc', val_accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
				self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
				
				# Calculate attn_softmax:
				if attn_list is not None:
					softmax_attn = nn.functional.softmax(attn_list, dim=1).cpu().detach().numpy()

					self.validation_step_outputs.append([embeddings, labels, val_loss, probs, metrics, views, softmax_attn, filenames])
					return embeddings, labels, val_loss, logits, metrics, views, softmax_attn, filenames

				else:
					self.validation_step_outputs.append([embeddings, labels, val_loss, probs, metrics, filenames])
					return embeddings, labels, val_loss, logits, metrics, filenames


			elif self.task == 'regression' or self.task == 'regression_mil':
				val_loss = self.regress_loss(Y_hat, labels)
				val_mae = self.mae(Y_hat, labels)
				val_mse = self.mse(Y_hat, labels)

				metrics = val_mae, val_mse
				# Log all metrics
				self.log('val_mae', val_mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
				self.log('val_mse', val_mse, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

				# Log losses
				self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

				if attn_list is not None:
					softmax_attn = nn.functional.softmax(attn_list, dim=1).cpu().detach().numpy()
					self.validation_step_outputs.append([embeddings, labels, val_loss, probs, metrics, views, softmax_attn, filenames])
					return embeddings, labels, val_loss, logits, metrics, softmax_attn, filenames
				else:
					self.validation_step_outputs.append([embeddings, labels, val_loss, probs, metrics, filenames])
					return embeddings, labels, val_loss, logits, metrics, filenames


	def on_validation_epoch_end(self):

		'''
		Concatenate all video embeddings into a single tensor 
		embeddings are visualzied with TSNE set with a fixed seed across experiments
		WARNING: THIS IS NOT DESIGNED TO WORK WITH DDP AND MULTIPLE GPU WORKERS YET
		'''
		outputs = self.validation_step_outputs

		if self.task == 'acdc': 
			self.embeddings = torch.cat([v[0] for v in outputs])
			self.labels = torch.cat([v[1] for v in outputs])

			# Embeddings are 512 dim vectors
			self.embeddings = np.array(self.embeddings.cpu())
			self.labels = np.array(self.labels.cpu())
			
			# t-SNE clustering
			tsne = UMAP(n_neighbors=10, random_state=12345)
			tsne_results = tsne.fit_transform(self.embeddings)
			labels = [np.where(r==1)[0][0] for r in self.labels]
			print('t-SNE clustering completed...')
			
			sns.set_style("white")
			plt.figure(figsize=(8,8))
			df = pd.DataFrame({'tsne-2d-one':tsne_results[:,0], 'tsne-2d-two':tsne_results[:,1], 'labels':labels})

			tsne_plot = sns.scatterplot(
				x="tsne-2d-one", y="tsne-2d-two",
				hue="labels",
				palette=sns.color_palette("hls", 5),
				data=df,
				legend="full",
				size="labels",
				alpha=1,
			)

			self.emb_plot = tsne_plot.get_figure()
			self.tsne_df = wandb.Table(dataframe=df)

			self.logger.experiment.log({'cmr_embedding': self.emb_plot, "gobal_step": self.global_step})
			self.logger.experiment.log({"tsne_exported": self.tsne_df})

		if self.task == 'classification' or self.task == 'classification_mil':
			self.log('val_auc_epoch', self.val_auroc.compute(), prog_bar=True, sync_dist=True)
			self.log('val_f1_epoch', self.val_f1.compute(), prog_bar=True, sync_dist=True)
			
			print(f'AUC: {self.val_auroc.compute()}')
			self.val_auroc.reset()
			self.val_f1.reset()

			### DEBUG SANITY CHECK FOR AUC ###
			labels = torch.cat([v[1] for v in outputs]).cpu().numpy()
			probs = torch.cat([v[3] for v in outputs]).cpu().numpy()
			views = list([v[5] for v in outputs])
			softmax_attn = list([v[6] for v in outputs])
			filenames = list([v[7] for v in outputs])

			df = pd.DataFrame(np.column_stack([probs, labels, filenames]), columns=('probs','labels','filenames'))
			print(df)

			attn_df = df
			attn_df['views'] = pd.Series(views)
			attn_df['softmax_attn'] = pd.Series(softmax_attn)
			# Needs pandas 1.13 or greater for this
			# Move to testing phase when I know it works
			attn_df = attn_df.explode(['views','softmax_attn'])
			attn_df = attn_df.explode(['views','softmax_attn'])

			self.logger.experiment.log({"preds_output": wandb.Table(dataframe=attn_df, dtype=AnyType)})			
			#self.logger.experiment.log({"preds_output": wandb.Table(dataframe=df)})			


		self.validation_step_outputs.clear()
		outputs.clear()

	def test_step(self, batch, batch_idx):
		'''
		Final forward pass through the test sets
		'''
		self.start_time = time.time()

		inputs, views, target, filenames = batch

		if 'regression_mil' in self.task:
			labels = torch.unsqueeze(target, dim=0).float()	

		elif 'classification_mil' in self.task:
			labels = torch.unsqueeze(target, dim=0).int()

		else:
			labels = torch.unsqueeze(target, dim=1).float()	

		# filenames, views, labels = target
		
		'''
		For now just spitting out embeddings, will change this to probs + embedding later
		'''
		# video_emb, attn_list = self.model(video, return_attentions=True)		
		# print('Exported attention maps from', len(attn_list), 'layers')

		logits, probs, Y_hat, embeddings, attn_maps = self.forward(inputs, generate_attnmaps=True)
		
		if self.task == 'classification' or self.task == 'classification_mil':
			test_loss = self.classif_loss(logits, labels.float())
			test_accuracy = self.accuracy(logits, labels)

			# AUC update; compute later and log later
			self.test_auroc.update(logits, labels)
			metrics = test_loss, test_accuracy 

			# Log non-modular metrics
			self.log('test_acc', test_accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

		elif self.task == 'regression' or 'regression_mil':
			test_loss = self.regress_loss(Y_hat, labels)
			test_mae = self.mae(Y_hat, labels)
			test_mse = self.mse(Y_hat, labels)

			metrics = test_mae, test_mse
			# Log all metrics
			self.log('test_mae', test_mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
			self.log('test_mse', test_mse, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

		# Log losses
		self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

		# if attn_list is not None:
		# 	softmax_attn = nn.functional.softmax(attn_list, dim=1).cpu().detach().numpy()
		# 	self.test_step_outputs.append([embeddings, labels, val_loss, probs, metrics, views, softmax_attn, filenames])
		# 	return embeddings, labels, val_loss, logits, metrics, softmax_attn, filenames
		
		
		# Custom logging line for conversion to mp4 (ffmpeg) and save with directory tree structure
		# Too expensive to put this step in epoch_end because of memory issues, unless this can be cached into tmp / RAM
		# Maybe add flag here on config file to control generation of attention maps
		if self.print_attention_maps:
			self._hdf5_writer(inputs, attn_maps, views, filenames)			
		
		self.test_step_outputs.append([embeddings, labels, test_loss, probs, metrics, filenames])
		return embeddings, labels, test_loss, Y_hat, metrics, filenames

	def on_test_epoch_end(self):
		'''
		Save predictions and probability matrix for plotting / troubleshooting
		Dataframe of format:
		{ Filename, Probs, Targets }
		'''
		print(f'Elapsed time for phase = "Testing": {round((time.time()-self.start_time), 2)}')
		outputs = self.test_step_outputs


		if self.task == 'classification' or self.task == 'classification_mil':
			self.log('test_auc_epoch', self.test_auroc.compute(), prog_bar=True, sync_dist=True)			
			print(f'AUC: {self.test_auroc.compute()}')
			self.test_auroc.reset()

			labels = torch.cat([v[1] for v in outputs]).cpu().numpy()
			probs = torch.cat([v[3] for v in outputs]).cpu().numpy()
			filenames = list([v[5] for v in outputs])
			## Add lines to maybe print out filenames and embeddings if needed ##

			df = pd.DataFrame(np.column_stack([filenames,probs, labels]), columns=('filenames','probs','labels'))
			print(df)
			self.logger.experiment.log({"final_test_output": wandb.Table(dataframe=df)})

		if self.task == 'regression' or self.task == 'regression_mil':
			self.embeddings = torch.cat([v[0] for v in outputs])
			self.labels = torch.cat([v[1] for v in outputs]).cpu().numpy()
			self.preds = torch.cat([v[3] for v in outputs]).cpu().numpy()
			self.filenames = list(v[5] for v in outputs)

			### DEBUG ###
			print(self.labels)
			print(self.preds)
			print(self.filenames)
			### DEBUG ###

			### TODO: Pandas dataframe generation still not working ###
			df = pd.DataFrame(np.column_stack([self.filenames, self.preds, self.labels]), columns=('filenames','preds','labels'))
			self.logger.experiment.log({"final_results": wandb.Table(dataframe=df, dtype=AnyType)})
		
	def configure_optimizers(self):
		'''
		Learning rates and optimizer configurations
		Currently configured to take optimizer = AdamW and lr_schduler = StepLR
		'''

		optimizer = instantiate_class(self.model.parameters(), self.optimizer_init)
		scheduler = instantiate_class(optimizer, self.lr_scheduler_init)

		return {"optimizer": optimizer, "lr_scheduler": scheduler}


	def _hdf5_writer(self, input_videos, attn_maps, views, filenames):
		'''
		Converts list of network attentions to np.arrays before saving as hdf5 files
		1. Iterate through attn_list, move to cpu and convert to np.array
		2. Save each layer as a separate dataset within one paraent hdf5 file
		'''
		if views is not None:
			# For UK BioBank type data (multi-instance & multi-view)
			for file_idx in range(len(filenames)):
				for view_idx, view in enumerate(views[file_idx]):
					os.makedirs(os.path.join(ATTN_DIR, view), exist_ok=True)

					### TODO: Need to create a way to handle SAX views; maybe save as separate hdf5 files ###
					if filenames[file_idx][:-3] == '.h5':
						output_filename = filenames[file_idx]
					else:
						output_filename = filenames[file_idx]+'.h5'

					counter = 00
					output_filepath = os.path.join(ATTN_DIR,view,output_filename[:-3]+'_'+counter+'.h5')

					if os.path.exists(output_filepath) == True:
						while os.path.exists(output_filepath):
							output_filepath = os.path.join(ATTN_DIR,view,output_filename[:-3]+'_'+counter+'.h5')
							counter += 1

					h5f = h5py.File(os.path.join(output_filepath), 'a')
					dset = h5f.create_dataset('input_video', data=input_videos[view_idx].cpu(), dtype='f')
					#print(f'Exported original input array: {input_videos[view_idx].shape}') 

					# Attention map is returned as a list = 16 with [3, nh, 25089, 393] for layer 1. (Dim=0 is = #views)
					for layer in range(len(attn_maps)):
						layer_attn = attn_maps[layer][view_idx]

						h5f = h5py.File(os.path.join(ATTN_DIR, view, output_filename), 'a') 
						dset = h5f.create_dataset('layer_%02d' %layer, data=layer_attn.cpu(), dtype='f')
		else:
			# For ACDC type data (single view)
			for file_idx in range(len(filenames)):
					os.makedirs(os.path.join(ATTN_DIR), exist_ok=True)
					if filenames[file_idx][:-3] == '.h5':
						output_filename = filenames[file_idx]
					else:
						output_filename = filenames[file_idx]+'.h5'
					h5f = h5py.File(os.path.join(ATTN_DIR, output_filename), 'a')
					dset = h5f.create_dataset('input_video', data=input_videos[file_idx].cpu(), dtype='f')

					# Attention map is returned as a list = 16 with [3, nh, 25089, 393] for layer 1. (Dim=0 is = #views)
					for layer in range(len(attn_maps)):
						layer_attn = attn_maps[layer][file_idx]

						h5f = h5py.File(os.path.join(ATTN_DIR, output_filename), 'a') 
						dset = h5f.create_dataset('layer_%02d' %layer, data=layer_attn.cpu(), dtype='f')
					#print(f'Exported attention map array for {filenames[file_idx]}-{view}')

# Might have to use custom BasePredictionWriter to save hdf5 files for visualizations
# Custom Lightning CLI class #

class Clinical_Trainer_CLI(LightningCLI):
	def add_arguments_to_parser(self, parser):
		parser.link_arguments("data.init_args.batch_size", "model.batch_size")
		parser.link_arguments("data.init_args.percentage", "model.data_percentage")
		parser.link_arguments("data.init_args.frames", "model.frames")
		parser.link_arguments("data.init_args.task", "model.task")
		parser.link_arguments("data.init_args.pos_weight", "model.pos_weight", apply_on="instantiate")
		parser.link_arguments("data.init_args.target", "model.target")
		
		# Optimizer and Scheduler instantiation
		parser.add_optimizer_args(
			torch.optim.AdamW,
			link_to="model.optimizer_init",
		)
		parser.add_lr_scheduler_args(
			torch.optim.lr_scheduler.StepLR,
			link_to="model.lr_scheduler_init",
		)

		parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_callback")
		parser.set_defaults({"checkpoint_callback.every_n_epochs": 100})

		parser.add_lightning_class_args(RichProgressBar, "rich_progressbar")
		parser.set_defaults({"rich_progressbar.refresh_rate": 10})

		# For tracking on wandb
		parser.link_arguments("optimizer.lr", "model.lr")
		parser.link_arguments("optimizer.weight_decay", "model.wd")


if __name__ == '__main__':
	'''
	The lightning CLI module configures the following:
		- Trainer (that now doesn't need to be defined)
		- Cardiac MRI Learner
		- MRI_Data_Module

	python mri_trainer.py --help {lists all exposed configurable variables}
	python mri_trainer.py fit --print_config {prints current defaults}
	python mri_trainer.py fit --config config.yaml {will run training loop with config.yaml settings}
	'''

	# Pass default trainer settings
	trainer_settings = {
			"precision": 32, 
			"accelerator": 'gpu',
			"devices": '1', 
			"strategy": 'ddp',
			"num_sanity_val_steps": 1, 
			"max_epochs": 100
			}


	# Supply different Data Module for downstream clinical tasks for easier readability
	cli = Clinical_Trainer_CLI(Cardiac_MRI_Net, trainer_defaults=trainer_settings, save_config_callback=None)

