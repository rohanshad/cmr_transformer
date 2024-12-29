'''
Visualize attention maps from vision transformer networks 
'''

import os
import glob
import sys
import argparse as ap
import cv2
import glob
import time
import multiprocessing

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import h5py
import numpy as np
import imageio_ffmpeg 
import subprocess
import math

#import model_factory
import random
import time

class VisualizeAttentions:
	def __init__(self, output_dir):
		self.output_dir = output_dir

	def start(self, input_hdf5):
		'''
		Parent function to generate mp4 videos for inputs and attention maps

		Structure of input_hdf5:
		filename_scan.h5
			├── input_video {data: 4d array [c, f, h, w]}
			├── layer_00 {data: 3d array} [1, 25089, 393]
			├── layer_01 {data: 3d array} [2, 6273, 1569]

		'''

		### DEBUG ###
		# Threads lock / hang after a while as you go deeper into layers 
		# Potentially expose the keys as a step that multiprocessing.pool can see
		# HDF5 can support multiple parallel reads but not writes/appends
		self.input_hdf5 = input_hdf5
		
		dat = h5py.File(self.input_hdf5, 'r')
		for i in dat.keys():
			print(i, "this is i")
			if i == 'input_video':
				self.render_original_video(np.array(dat[i]))
			else:
				self.generate_video_mp4(dat[i], 'magma', layer_num=i[-2:])
				print('Exported video for', i, 'from', os.path.basename(self.input_hdf5))

	def render_original_video(self, array):
		'''
		Generates a grayscale mp4 file of the input video using ffmpeg
		'''
		array = array.transpose(1, 0, 2, 3)
		save_dir = os.path.join(self.output_dir, os.path.basename(self.input_hdf5[:-3]), 'input_video', 'attention_frames')
		os.makedirs(save_dir, exist_ok = True)

		for i in range(array.shape[0]):
			plt.imshow(array[i][1,:,:]/225, cmap='gist_gray')
			plt.axis('off')
			plt.savefig(os.path.join(save_dir, 'frame%02d.png' %i), bbox_inches='tight')		

		input_filename = os.path.join(save_dir, 'frame%02d.png')
		output_filename = os.path.join(self.output_dir, os.path.basename(self.input_hdf5)[:-3],'input_video.mp4')
			
		command = 'ffmpeg -i "{input_filename}" ' \
					'-loglevel error -c:v libx264 -c:a copy ' \
					'-vf scale=224:-2 '\
					'-r 20 -pix_fmt yuv420p ' \
					'"{output_filename}"'.format(
						input_filename=input_filename,
						output_filename=output_filename
					)
		print(command)
		subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

		self.checkpoint_time = time.time()
		print(f'Rendered original video: {self.checkpoint_time}')

	def generate_attention_maps(self, array):
		'''
		Converts attention array into heatmap that can be visualized
		
		Reference:
		MVIT {layer 1 array shape}: [nh, 25089, 393]
		MVIT {layer 2 array shape}: [nh, 6273, 1569]
		MVIT {layer 3 array shape}: [nh, 6273, 393]
		'''
		print(array.shape)
		nh = array.shape[0]
		attentions = array[:, 1:, 0].reshape(nh, -1)

		## WARNING: THIS IS SPECIFIC FOR THE mvit_base_16x4 ARCHITECTURE ##
		f_featmap = 8
		w_featmap = int(math.sqrt(attentions.shape[1] // f_featmap))
		h_featmap = int(math.sqrt(attentions.shape[1] // f_featmap))

		# Reshape array to featmap dims, and then interpolate to full size video for viz
		attentions = torch.Tensor(attentions)
		attentions = attentions.reshape(nh, f_featmap, h_featmap, w_featmap)
		attentions = (nn.functional.interpolate(attentions.unsqueeze(0), size=(16,224,224), mode='nearest')[0].cpu().numpy())

		return attentions

	def generate_video_mp4(self, array, color_map, layer_num):
		'''
		Generates a mp4 video from input arrays using ffmpeg
		Retains magma color scheme because it looks dope
		'''
		
		# attn_map shape: [nh, 16, 224, 224]
		attn_map = self.generate_attention_maps(array)
		nh = attn_map.shape[0]
		print(f'Generated attention maps: {round(time.time() - self.checkpoint_time, 2)}')

		
		for h in range(nh):
			print('Processing from head:', h)
			save_dir = os.path.join(self.output_dir, os.path.basename(self.input_hdf5[:-3]), 'layer_%02d' %int(layer_num), 'attention_frames', 'head_%02d' %int(h))
			os.makedirs(save_dir, exist_ok = True)
			array = attn_map[h]
			g = plt.imshow(array[0]/225, cmap=color_map)
			plt.axis('off')
			plt.savefig(os.path.join(save_dir,'frame%02d.png' %0), bbox_inches='tight')            
			for i in range(1, array.shape[0]):
				g.set_data(array[i]/225)
				g.autoscale()
				# Might keep axis but remove bbox 
				plt.savefig(os.path.join(save_dir,'frame%02d.png' %i), bbox_inches='tight')	
			
			input_filename = os.path.join(save_dir, 'frame%02d.png')
			print(input_filename)
			output_filename = os.path.join(self.output_dir, os.path.basename(self.input_hdf5)[:-3],'layer_%02d' %int(layer_num)+'_head_%02d' %int(h) +'.mp4')
			
			print(f'Generating pngs for head {nh}: {round(time.time() - self.checkpoint_time, 2)}')
			plt.clf()

			command = 'ffmpeg -i "{input_filename}" ' \
						'-loglevel error -c:v libx264 -c:a copy ' \
						'-vf scale=224:-2 '\
						'-r 20 -pix_fmt yuv420p ' \
						'"{output_filename}"'.format(
							input_filename=input_filename,
							output_filename=output_filename
						)
			print(f'ffmpeg for head {nh}: {round(time.time() - self.checkpoint_time, 2)}')
			print(command)
			subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)




if __name__ == '__main__':
#	assert os.listdir('C:/Users/matthewleipzig/Documents/CT_Surgery/rohan/coding/output')
	parser = ap.ArgumentParser(
		description="Generates attention heatmaps",
		epilog="Version 0.1; Created by Rohan Shad, MD"
	)

	parser.add_argument('-i', '--input_dir', metavar='', required=False, help='Full path to root directory', default='/Users/matthewleipzig/Downloads/layer_single_test')
	parser.add_argument('-o', '--output_dir', metavar='', required=False, default='/Users/matthewleipzig/Downloads/output_layer_single_test', help='View to plot')
	parser.add_argument('-c', '--cpus', metavar='', required=False, type=int, default=10, help='number of cores to use in multiprocessing')


	args = vars(parser.parse_args())
	#print(args

	input_dir = args['input_dir']
	output_dir = args['output_dir']
	cpus = args['cpus']

	filenames =  glob.glob(os.path.join(input_dir, '*h5'))
	#random.shuffle(filenames)
	p = multiprocessing.Pool(processes=cpus)
	v = VisualizeAttentions(output_dir)

	start_time = time.time()
	for f in filenames:
		if f[-2:] == 'h5':
			if cpus > 1:
				p.apply_async(v.start, [f])
			else:
				v.start(f)

	p.close()
	p.join()

	print('Elapsed time:', round((time.time() - start_time), 2))




