'''
Model factory script for public release
'''

from transformers import AutoModel
import torch
from torch import nn
from arch import resnet, mvit
from pyaml_env import BaseConfig, parse_config
import platform

# Read local_config.yaml for local variables 
device = platform.uname().node.replace('-','_')
cfg = BaseConfig(parse_config('local_config.yaml'))
if 'sh' in device:
	device = 'sherlock'
elif '211' in device:
	device = 'cubic'
try:
	PRETRAIN_WEIGHTS = getattr(cfg, device).pretrain_dir
except:
	print('------------------------------------')
	print(f"WARNING: local_config.yaml is not configured correctly, please see README")
	print(f"Unable to load Kinetics pretrained checkpoint for mvit")
	print('------------------------------------')

def resnet_builder(model_depth: int=50, pretrained:bool=True, n_classes: int=700):
	'''
	Resnets with pretrained weights from: https://github.com/kenshohara/3D-ResNets-PyTorch
	Dynamically build 3d resnets with 50, 101, 152, 200 layers. Pretrained on Kinetics-700 for 200 epochs

	Weights available: https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4
	'''

	model = resnet.generate_model(model_depth=model_depth, n_classes=700)

	if pretrained is True:
		print('Loading Kinetics-700 pretrained weights for', model_depth, 'layer 3d resnet...')
		pretrained_model = torch.load(PRETRAIN_WEIGHTS+'/r3d'+str(model_depth)+'_K_200ep.pth', map_location='cpu')
		model.load_state_dict(pretrained_model['state_dict'])

	# Return dimensions of classifier head input 
	dim_feats = model.fc.in_features

	# Finetuning setup to remove classification head
	new_output = nn.Identity()
	model.fc = new_output

	return model, dim_feats


def mri_mvit(frames: int=16, kinetics_pretrained: bool=True, n_first_layers: int=0):
	'''
	Constructs a multiscale vision transformer using pytorchvideo from: https://arxiv.org/abs/2104.11227
	https://github.com/facebookresearch/pytorchvideo 

	Uses Kinetics-400 pretrained weights for initialization by default
	'''
	print('Initializing mvit with kinetics_pretrained set to', str(kinetics_pretrained))
	if frames == 16:
		model = mvit.mvit_base_16x4(pretrained=kinetics_pretrained)
	elif frames == 32:
		model = mvit.mvit_base_32x3(pretrained=kinetics_pretrained)

	# Freeze first n layers of network
	if n_first_layers > 0:
		for idx in range(n_first_layers):
			for param in list(model.blocks[idx].parameters()):
				param.requires_grad = False
	print(f'Freezing first {n_first_layers} layers of mvit')			

	# Return dimensions of classifier head input 
	dim_feats = model.head.proj.in_features

	# Finetuning setup to remove classification head
	new_output = nn.Identity()
	model.head.proj = new_output

	return model, dim_feats


#### Network Modules for downstream inference ####

class Cardiac_MRI_Encoder(nn.Module):
	'''
	Downstream task module for finetuning constrastive pretrained models
	Multi-GPU should be handled via pl.lightning trainer. Linear and Finetuning settings.
	'''
	def __init__(self, video_model, video_model_depth, proj_dim, proj_layers, output_classes, frames, mode, pretrained):
		super().__init__()

		# Load specific architectures
		if video_model == 'resnet':
			print('Loading ResNet with', video_model_depth, 'layers...')
			self.video_encoder, self.video_dim = resnet_builder(model_depth=video_model_depth, pretrained=pretrained, n_classes=700)

		elif video_model == 'mvit':
			print('Loading multiscale vision transformer...')
			self.video_encoder, self.video_dim = mri_mvit(frames=frames, kinetics_pretrained=pretrained, n_first_layers=0)

		else:
			print('Undefined architecture')

		# Build projection layers and classification head:
		self.video_proj = make_projection_layers(self.video_dim, proj_dim, proj_layers)
		#self.video_fc = MultiLabelBinaryFCLayer(512, output_classes, dropout=None)

		# Evaluation phase settings	
		if mode == 'linear':
			print('Freezing encoder and projection layers for zero shot evaluation')
			for param in list(self.video_encoder.parameters()):
				param.requires_grad = False
			for param in list(self.video_proj.parameters()):
				param.requires_grad = False

		elif mode == 'finetune': 
			print('Freezing encoder only for finetuning')
			for param in list(self.video_encoder.parameters()):
				param.requires_grad = False
			for param in list(self.video_proj.parameters()):
				param.requires_grad = True

		elif mode == 'transfer':
			print('Unfreeze encoder and projection layers for transfer learning')
			for param in list(self.video_encoder.parameters()):
				param.requires_grad = True
			for param in list(self.video_proj.parameters()):
				param.requires_grad = True

		else:
			print('Not yet defined')

	def forward(self, video, return_attentions=False, return_logits=False):
		
		if return_attentions == True:
			video_v, attn_map_list = self.video_encoder(video, return_attentions=True)
		else: 
			video_v = self.video_encoder(video)

		# Apply MLP projection to network output embeddings
		video_v = self.video_proj(video_v)

		#Remove this bit if need be, but this should be harmless to normalize
		video_v = nn.functional.normalize(video_v, dim=1)

		# Return video embedding
		if return_logits == False:
			if return_attentions == True:
				return video_v, attn_map_list
			else:
				return video_v
		else:
			print("Haven't built this yet")


class Attn_Net_Gated(nn.Module):
	'''
	Gated self attention module modified from DeepAttention MIL
	NOTE: Requires batch_size = 1 to work with varying bags of videos
	Attention-based Deep Multiple Instance Learning (arXiv preprint arXiv:1802.04712)
	Pytorch implementation: https://github.com/AMLab-Amsterdam/AttentionDeepMIL

	Added optional layernorm
	'''
	
	def __init__(self, L = 512, D = 256, dropout = False, n_classes = 1, normalize=False):
		super().__init__()
		self.normalize = normalize
		if self.normalize == True:
			self.layernorm = torch.nn.LayerNorm(L)

		self.attention_a = [
			nn.Linear(L, D),
			nn.Tanh()]

		self.attention_b = [nn.Linear(L, D),
							nn.Sigmoid()]
		if dropout:
			self.attention_a.append(nn.Dropout(0.25))
			self.attention_b.append(nn.Dropout(0.25))

		self.attention_a = nn.Sequential(*self.attention_a)
		self.attention_b = nn.Sequential(*self.attention_b)

		self.attention_c = nn.Linear(D, n_classes)

	def forward(self, x):
		if self.normalize == True:
			x = self.layernorm(x)
		a = self.attention_a(x)
		b = self.attention_b(x)
		A = a.mul(b)
		A = self.attention_c(A)  # n_classes x N
		return A, x


class Multiview_Attention(nn.Module):
	'''
	Multi-instance gated attention module for aggregating information from multiple MRI
	views independent of the number of views available for each patient.

	Input is a tensor of size [arbitrary num_views x Projection Dimension] 
	Incorporates multi-label binary fc classification head 

	'''
	def __init__(self, proj_dim=512, dropout=False, n_classes=1, task='regression_mil', normalize=False):
		super().__init__()
		
		self.attention_net = nn.Sequential(
			nn.Linear(proj_dim, proj_dim//2), nn.ReLU(),
			nn.Dropout(0.25),
			nn.Linear(proj_dim//2, proj_dim//2), nn.ReLU(),
			nn.Dropout(0.25),
			Attn_Net_Gated(L = proj_dim//2, D = proj_dim//4, dropout = dropout, n_classes = n_classes, normalize=normalize)
			)
		self.task = task
		self.debug = False

		# MultiLabelBinary Classifier can handle anything from n = 1 to n > 1 classes
		self.classifier_head = MultiLabelBinaryFCLayer(proj_dim//2, n_classes, dropout=0.25, task=self.task)
		init_model(self)
		if self.task == 'regression_mil':
			# Hardcoded bias term for LVEF regression
			self.classifier_head.fc[0].bias.data[0] = 0.57


	def forward(self, emb_stack, task, return_features=False, attention_only=False):
		
		A, emb_stack = self.attention_net(emb_stack)
		A = torch.transpose(A, 1, 0)
		if attention_only:
			return A
		A_raw = A
		
		A = nn.functional.softmax(A, dim=1)
		M = torch.mm(A, emb_stack)

		if self.debug:
			print(f'A: {A.shape}')
			print(f'emb_stack: {emb_stack.shape}')
			print(f'M: {M.shape}')

		logits, _, _, _, _ = self.classifier_head(M)
		
		if self.task == 'regression_mil':
			Y_prob = None
			Y_hat = torch.mul(logits, 100)

			if self.debug:
				print(f'PREDS: {Y_hat}')

		elif self.task == 'classification_mil':
			Y_prob = torch.sigmoid(logits)
			Y_hat = (Y_prob > 0.5).int()

			if self.debug:
				print(f'PROBS: {Y_prob}')
				print(f'PREDS: {Y_hat}')

		results_dict = {}
		if return_features:
			results_dict.update({'features': M})

		return logits, Y_prob, Y_hat, A_raw, results_dict

class MultiLabelBinaryFCLayer(nn.Module):
	'''
	A multi-label FC classifier layer with `num_label` output heads and each a single output for each class
	The output size is a vector of shape = M (shape equals num_classes) from the self-attention module above. 
	This is to ensure the output layer has logits of length = num of classes. 
	'''
	def __init__(self, proj_dim=512, n_classes=1, dropout=0.25, task='regression_mil'):
		super().__init__()
		
		self.fc = nn.Sequential(nn.Linear(proj_dim, 1))
		self.proj_dim = proj_dim
		self.out_features = n_classes
		self.task = task
		self.debug = False

		if dropout is not None and dropout > 0:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None
		init_model(self)

	def forward(self, x, task=None):
		if self.dropout is not None:
			logits = self.fc(self.dropout(x))
		else:
			logits = self.fc(x)

		if self.task == 'regression' or self.task == 'regression_mil':
			Y_hat = torch.mul(logits, 100)
			Y_prob = None

		elif self.task == 'classification_mil':
			logits = logits.float()

			if self.debug:
				print(f'LOGITS SHAPE (WITHIN MF): {logits.shape}')
				print(f'LOGITS (WITHIN MF): {logits}')

			Y_prob = None
			Y_hat = None

		results_dict = None
		A_raw = None

		return logits, Y_prob, Y_hat, A_raw, results_dict


##############################     Helper Functions from Yuhao Zhang  ##############################

def pool(h, mask, type='max'):
	'''
	Define pooling type for BERT
	'''

	# Need to set infinity number here #
	if type == 'max':
		h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
		return torch.max(h, 1)[0]
	elif type == 'mean':
		h = h.masked_fill(mask, 0)
		return h.sum(1) / (mask.size(1) - mask.float().sum(1))
	elif type == 'sum':
		h = h.masked_fill(mask, 0)
		return h.sum(1)
	else:
		raise Exception('Unsupported pooling type: ' + type)


def make_projection_layers(in_dim, out_dim, num_layers=2, dropout_layer=None):
	'''
	Generates projection layers for arbitrary in/out dims
	'''
	
	# Simple linear layer
	if num_layers == 1: 
		return nn.Linear(in_dim, out_dim)
	else:
	# Multilayer Perceptron type stuff?
		layers = []
		for i in range(num_layers-1):
			layers += [
				nn.Linear(in_dim, in_dim),
				nn.ReLU()
			]
			if dropout_layer:
				layers += [dropout_layer]
		layers += [nn.Linear(in_dim, out_dim)]
		return nn.Sequential(*layers)

def load_contrastive_pretrained_weights(input_model, checkpoint_path):
	'''
	Loads only necessary key, value pairs from supplied state_dict for video_encoder
	'''

	model = input_model
	model_dict = model.state_dict()

	pretrained = torch.load(checkpoint_path, map_location='cpu')
	pretrained_state_dict =  pretrained['state_dict']

	try:
		# 1. filter out unnecessary keys
		video_encoder_dict = {k.strip('model.'): v for k, v in pretrained_state_dict.items() if k.strip('model.') in model_dict}
		# 2. overwrite entries in the existing state dict
		model_dict.update(video_encoder_dict) 
		# 3. load the new state dict
		model.load_state_dict(model_dict)

		print('Successfully loaded weights!')

	except Exception as ex:
		print(ex)
		
	return model

def load_finetuned_mri_network_checkpoints(video_encoder, classifier_head, checkpoint_path):
	'''
	Loads weights for classifier head
	'''

	video_encoder_dict = video_encoder.state_dict()
	classifier_head_dict = classifier_head.state_dict()

	pretrained = torch.load(checkpoint_path, map_location='cpu')
	pretrained_state_dict =  pretrained['state_dict']

	try:
		# 1. Load weights onto video and classifier models
		pretrained_state_dict = {k.strip('model.'): v for k, v in pretrained_state_dict.items()}

		video_encoder_dict.update(pretrained_state_dict)
		classifier_head_dict.update(pretrained_state_dict)
 
		# 2. load the new state dict
		video_encoder.load_state_dict(video_encoder_dict, strict=False)
		classifier_head.load_state_dict(classifier_head_dict, strict=False)

		print('Successfully loaded weights!')

	except Exception as ex:
		print(ex)
		
	return video_encoder, classifier_head


def init_model(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()

		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)


