'''
Implementation of mvit from pytorchvideo rearranged to load in a single script:
Multiscale Vision Transformers
Haoqi Fan et. al. https://arxiv.org/abs/2104.11227

1. layers.utils
2. layers.attention
3. models.head.create_vit_basic_head
4. models.weight_init
5. models.stem
6. vision_transformers.py
'''


from functools import partial
import math
from typing import Callable, List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from fvcore.nn.weight_init import c2_msra_fill

import numpy
import torch
import torch.nn as nn
from torch.nn.common_types import _size_3_t, _size_2_t

#from .drop_path import DropPath


MODEL_ZOO_ROOT_DIR = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo"

class MultiscaleVisionTransformers(nn.Module):
	'''
	Multiscale Vision Transformers
	Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik,
	Christoph Feichtenhofer
	https://arxiv.org/abs/2104.11227
	'''
	def __init__(
		self,
		*,
		patch_embed: Optional[nn.Module],
		cls_positional_encoding: nn.Module,
		pos_drop: Optional[nn.Module],
		norm_patch_embed: Optional[nn.Module],
		blocks: nn.ModuleList,
		norm_embed: Optional[nn.Module],
		head: Optional[nn.Module],
	) -> None:
		"""
		Args:
			patch_embed (nn.Module): Patch embed module.
			cls_positional_encoding (nn.Module): Positional encoding module.
			pos_drop (Optional[nn.Module]): Dropout module after patch embed.
			blocks (nn.ModuleList): Stack of multi-scale transformer blocks.
			norm_layer (nn.Module): Normalization layer before head.
			head (Optional[nn.Module]): Head module.
		"""
		super().__init__()
		set_attributes(self, locals())
		assert hasattr(
			cls_positional_encoding, "patch_embed_shape"
		), "cls_positional_encoding should have attribute patch_embed_shape."
		init_net_weights(self, init_std=0.02, style="vit")

	def forward(self, x: torch.Tensor, return_attentions: bool=False) -> torch.Tensor:
		if self.patch_embed is not None:
			x = self.patch_embed(x)
		x = self.cls_positional_encoding(x)

		if self.pos_drop is not None:
			x = self.pos_drop(x)

		if self.norm_patch_embed is not None:
			x = self.norm_patch_embed(x)

		thw = self.cls_positional_encoding.patch_embed_shape
		attn_map_list = []
		for blk in self.blocks:
			if return_attentions == True:
				x, thw, attn_map_array = blk(x, thw, return_attentions)
				attn_map_list.append(attn_map_array)
				#attn_map = torch.cat(attn_map_list, dim=0)
			else:
				x, thw = blk(x, thw)
		if self.norm_embed is not None:
			x = self.norm_embed(x)
		if self.head is not None:
			x = self.head(x)

		if return_attentions == True:
			return x, attn_map_list
		else:
			return x

def create_vit_basic_head(
	*,
	# Projection configs.
	in_features: int,
	out_features: int,
	# Pooling configs.
	seq_pool_type: str = "cls",
	# Dropout configs.
	dropout_rate: float = 0.5,
	# Activation configs.
	activation: Callable = None,
	) -> nn.Module:
	"""
	Creates vision transformer basic head.
	::
										Pooling
										   ↓
										Dropout
										   ↓
									   Projection
										   ↓
									   Activation
	Activation examples include: ReLU, Softmax, Sigmoid, and None.
	Pool type examples include: cls, mean and none.
	Args:
		in_features: input channel size of the resnet head.
		out_features: output channel size of the resnet head.
		pool_type (str): Pooling type. It supports "cls", "mean " and "none". If set to
			"cls", it assumes the first element in the input is the cls token and
			returns it. If set to "mean", it returns the mean of the entire sequence.
		activation (callable): a callable that constructs vision transformer head
			activation layer, examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and
			None (not applying activation).
		dropout_rate (float): dropout rate.
	"""
	assert seq_pool_type in ["cls", "meam", "none"]

	if seq_pool_type in ["cls", "mean"]:
		seq_pool_model = SequencePool(seq_pool_type)
	elif seq_pool_type == "none":
		seq_pool_model = None
	else:
		raise NotImplementedError

	if activation is None:
		activation_model = None
	elif activation == nn.Softmax:
		activation_model = activation(dim=1)
	else:
		activation_model = activation()

	return VisionTransformerBasicHead(
		sequence_pool=seq_pool_model,
		dropout=nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None,
		proj=nn.Linear(in_features, out_features),
		activation=activation_model,
	)


def create_multiscale_vision_transformers(
	*,
	spatial_size: _size_2_t,
	temporal_size: int,
	cls_embed_on: bool = True,
	sep_pos_embed: bool = True,
	depth: int = 16,
	norm: str = "layernorm",
	# Patch embed config.
	enable_patch_embed: bool = True,
	input_channels: int = 3,
	patch_embed_dim: int = 96,
	conv_patch_embed_kernel: Tuple[int] = (3, 7, 7),
	conv_patch_embed_stride: Tuple[int] = (2, 4, 4),
	conv_patch_embed_padding: Tuple[int] = (1, 3, 3),
	enable_patch_embed_norm: bool = False,
	use_2d_patch: bool = False,
	# Attention block config.
	num_heads: int = 1,
	mlp_ratio: float = 4.0,
	qkv_bias: bool = True,
	dropout_rate_block: float = 0.0,
	droppath_rate_block: float = 0.0,
	pooling_mode: str = "conv",
	pool_first: bool = False,
	embed_dim_mul: Optional[List[List[int]]] = None,
	atten_head_mul: Optional[List[List[int]]] = None,
	pool_q_stride_size: Optional[List[List[int]]] = None,
	pool_kv_stride_size: Optional[List[List[int]]] = None,
	pool_kv_stride_adaptive: Optional[_size_3_t] = None,
	pool_kvq_kernel: Optional[_size_3_t] = None,
	# Head config.
	head: Optional[Callable] = create_vit_basic_head,
	head_dropout_rate: float = 0.5,
	head_activation: Callable = None,
	head_num_classes: int = 400,
	) -> nn.Module:
	'''
	Build Multiscale Vision Transformers (MViT) for recognition. A Vision Transformer
	(ViT) is a specific case of MViT that only uses a single scale attention block.

	Args:
		spatial_size (_size_2_t): Input video spatial resolution (H, W). If a single
			int is given, it assumes the width and the height are the same.
		temporal_size (int): Number of frames in the input video.
		cls_embed_on (bool): If True, use cls embed in the model. Otherwise features
			are average pooled before going to the final classifier.
		sep_pos_embed (bool): If True, perform separate spatiotemporal embedding.
		depth (int): The depth of the model.
		norm (str): Normalization layer. It currently supports "layernorm".

		enable_patch_embed (bool): If true, patchify the input video. If false, it
			assumes the input should have the feature dimension of patch_embed_dim.
		input_channels (int): Channel dimension of the input video.
		patch_embed_dim (int): Embedding dimension after patchifing the video input.
		conv_patch_embed_kernel (Tuple[int]): Kernel size of the convolution for
			patchifing the video input.
		conv_patch_embed_stride (Tuple[int]): Stride size of the convolution for
			patchifing the video input.
		conv_patch_embed_padding (Tuple[int]): Padding size of the convolution for
			patchifing the video input.
		enable_patch_embed_norm (bool): If True, apply normalization after patchifing
			the video input.
		use_2d_patch (bool): If True, use 2D convolutions to get patch embed.
			Otherwise, use 3D convolutions.

		num_heads (int): Number of heads in the first transformer block.
		mlp_ratio (float): Mlp ratio which controls the feature dimension in the
			hidden layer of the Mlp block.
		qkv_bias (bool): If set to False, the qkv layer will not learn an additive
			bias. Default: False.
		dropout_rate_block (float): Dropout rate for the attention block.
		droppath_rate_block (float): Droppath rate for the attention block.
		pooling_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
			(average pooling), and "max" (max pooling).
		pool_first (bool): If set to True, pool is applied before qkv projection.
			Otherwise, pool is applied after qkv projection. Default: False.
		embed_dim_mul (Optional[List[List[int]]]): Dimension multiplication at layer i.
			If X is used, then the next block will increase the embed dimension by X
			times. Format: [depth_i, mul_dim_ratio].
		atten_head_mul (Optional[List[List[int]]]): Head dimension multiplication at
			layer i. If X is used, then the next block will increase the head by
			X times. Format: [depth_i, mul_dim_ratio].
		pool_q_stride_size (Optional[List[List[int]]]): List of stride sizes for the
			pool q at each layer. Format:
			[[i, stride_t_i, stride_h_i, stride_w_i], ...,].
		pool_kv_stride_size (Optional[List[List[int]]]): List of stride sizes for the
			pool kv at each layer. Format:
			[[i, stride_t_i, stride_h_i, stride_w_i], ...,].
		pool_kv_stride_adaptive (Optional[_size_3_t]): Initial kv stride size for the
			first block. The stride size will be further reduced at the layer where q
			is pooled with the ratio of the stride of q pooling. If
			pool_kv_stride_adaptive is set, then pool_kv_stride_size should be none.
		pool_kvq_kernel (Optional[_size_3_t]): Pooling kernel size for q and kv. It None,
			the kernel_size is [s + 1 if s > 1 else s for s in stride_size].

		head (Callable): Head model.
		head_dropout_rate (float): Dropout rate in the head.
		head_activation (Callable): Activation in the head.
		head_num_classes (int): Number of classes in the final classification head.
	'''

	if use_2d_patch:
		assert temporal_size == 1, "If use_2d_patch, temporal_size needs to be 1."
	if pool_kv_stride_adaptive is not None:
		assert (
			pool_kv_stride_size is None
		), "pool_kv_stride_size should be none if pool_kv_stride_adaptive is set."
	if norm == "layernorm":
		norm_layer = partial(nn.LayerNorm, eps=1e-6)
	else:
		raise NotImplementedError("Only supports layernorm.")
	if isinstance(spatial_size, int):
		spatial_size = (spatial_size, spatial_size)

	conv_patch_op = nn.Conv2d if use_2d_patch else nn.Conv3d

	patch_embed = (
		create_conv_patch_embed(
			in_channels=input_channels,
			out_channels=patch_embed_dim,
			conv_kernel_size=conv_patch_embed_kernel,
			conv_stride=conv_patch_embed_stride,
			conv_padding=conv_patch_embed_padding,
			conv=conv_patch_op,
		)
		if enable_patch_embed
		else None
	)

	input_dims = [temporal_size, spatial_size[0], spatial_size[1]]
	input_stirde = (
		(1,) + tuple(conv_patch_embed_stride)
		if use_2d_patch
		else conv_patch_embed_stride
	)

	patch_embed_shape = (
		[input_dims[i] // input_stirde[i] for i in range(len(input_dims))]
		if enable_patch_embed
		else input_dims
	)

	cls_positional_encoding = SpatioTemporalClsPositionalEncoding(
		embed_dim=patch_embed_dim,
		patch_embed_shape=patch_embed_shape,
		sep_pos_embed=sep_pos_embed,
		has_cls=cls_embed_on,
	)

	dpr = [
		x.item() for x in torch.linspace(0, droppath_rate_block, depth)
	]  # stochastic depth decay rule

	if dropout_rate_block > 0.0:
		pos_drop = nn.Dropout(p=dropout_rate_block)

	dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
	if embed_dim_mul is not None:
		for i in range(len(embed_dim_mul)):
			dim_mul[embed_dim_mul[i][0]] = embed_dim_mul[i][1]
	if atten_head_mul is not None:
		for i in range(len(atten_head_mul)):
			head_mul[atten_head_mul[i][0]] = atten_head_mul[i][1]

	norm_patch_embed = norm_layer(patch_embed_dim) if enable_patch_embed_norm else None

	mvit_blocks = nn.ModuleList()

	pool_q = [[] for i in range(depth)]
	pool_kv = [[] for i in range(depth)]
	stride_q = [[] for i in range(depth)]
	stride_kv = [[] for i in range(depth)]

	if pool_q_stride_size is not None:
		for i in range(len(pool_q_stride_size)):
			stride_q[pool_q_stride_size[i][0]] = pool_q_stride_size[i][1:]
			if pool_kvq_kernel is not None:
				pool_q[pool_q_stride_size[i][0]] = pool_kvq_kernel
			else:
				pool_q[pool_q_stride_size[i][0]] = [
					s + 1 if s > 1 else s for s in pool_q_stride_size[i][1:]
				]

	# If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
	if pool_kv_stride_adaptive is not None:
		_stride_kv = pool_kv_stride_adaptive
		pool_kv_stride_size = []
		for i in range(depth):
			if len(stride_q[i]) > 0:
				_stride_kv = [
					max(_stride_kv[d] // stride_q[i][d], 1)
					for d in range(len(_stride_kv))
				]
			pool_kv_stride_size.append([i] + _stride_kv)

	if pool_kv_stride_size is not None:
		for i in range(len(pool_kv_stride_size)):
			stride_kv[pool_kv_stride_size[i][0]] = pool_kv_stride_size[i][1:]
			if pool_kvq_kernel is not None:
				pool_kv[pool_kv_stride_size[i][0]] = pool_kvq_kernel
			else:
				pool_kv[pool_kv_stride_size[i][0]] = [
					s + 1 if s > 1 else s for s in pool_kv_stride_size[i][1:]
				]

	for i in range(depth):
		num_heads = round_width(num_heads, head_mul[i], min_width=1, divisor=1)
		patch_embed_dim = round_width(patch_embed_dim, dim_mul[i], divisor=num_heads)
		dim_out = round_width(
			patch_embed_dim,
			dim_mul[i + 1],
			divisor=round_width(num_heads, head_mul[i + 1]),
		)

		mvit_blocks.append(
			MultiScaleBlock(
				dim=patch_embed_dim,
				dim_out=dim_out,
				num_heads=num_heads,
				mlp_ratio=mlp_ratio,
				qkv_bias=qkv_bias,
				dropout_rate=dropout_rate_block,
				droppath_rate=dpr[i],
				norm_layer=norm_layer,
				kernel_q=pool_q[i],
				kernel_kv=pool_kv[i],
				stride_q=stride_q[i],
				stride_kv=stride_kv[i],
				pool_mode=pooling_mode,
				has_cls_embed=cls_embed_on,
				pool_first=pool_first,
			)
		)

	embed_dim = dim_out
	norm_embed = norm_layer(embed_dim)
	if head is not None:
		head_model = head(
			in_features=embed_dim,
			out_features=head_num_classes,
			seq_pool_type="cls" if cls_embed_on else "mean",
			dropout_rate=head_dropout_rate,
			activation=head_activation,
		)
	else:
		head_model = None

	return MultiscaleVisionTransformers(
		patch_embed=patch_embed,
		cls_positional_encoding=cls_positional_encoding,
		pos_drop=pos_drop if dropout_rate_block > 0.0 else None,
		norm_patch_embed=norm_patch_embed,
		blocks=mvit_blocks,
		norm_embed=norm_embed,
		head=head_model,
	)


#### POSITIONAL ENCODING ####
class PositionalEncoding(nn.Module):
	"""
	Applies a positional encoding to a tensor with shape (batch_size x seq_len x embed_dim).
	The positional encoding is computed as follows:
		PE(pos,2i) = sin(pos/10000^(2i/dmodel))
		PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))
		where pos = position, pos in [0, seq_len)
		dmodel = data embedding dimension = embed_dim
		i = dimension index, i in [0, embed_dim)
	Reference: "Attention Is All You Need" https://arxiv.org/abs/1706.03762
	Implementation Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
	"""

	def __init__(self, embed_dim: int, seq_len: int = 1024) -> None:
		super().__init__()
		pe = torch.zeros(seq_len, embed_dim, dtype=torch.float)
		position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, embed_dim, 2).float() * (-(math.log(10000.0)) / embed_dim)
		)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer("pe", pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		assert self.pe.size(1) >= x.size(1), (
			"Cannot apply position encoding of size "
			+ f"{self.pe.size()} when input has size {x.size()}"
		)
		return x + self.pe[:, : x.size(1), :]


class SpatioTemporalClsPositionalEncoding(nn.Module):
	"""
	Add a cls token and apply a spatiotemporal encoding to a tensor.
	"""

	def __init__(
		self,
		embed_dim: int,
		patch_embed_shape: Tuple[int, int, int],
		sep_pos_embed: bool = False,
		has_cls: bool = True,
	) -> None:
		"""
		Args:
			embed_dim (int): Embedding dimension for input sequence.
			patch_embed_shape (Tuple): The number of patches in each dimension
				(T, H, W) after patch embedding.
			sep_pos_embed (bool): If set to true, one positional encoding is used for
				spatial patches and another positional encoding is used for temporal
				sequence. Otherwise, only one positional encoding is used for all the
				patches.
			has_cls (bool): If set to true, a cls token is added in the beginning of each
				input sequence.
		"""
		super().__init__()
		assert (
			len(patch_embed_shape) == 3
		), "Patch_embed_shape should be in the form of (T, H, W)."
		self.cls_embed_on = has_cls
		self.sep_pos_embed = sep_pos_embed
		self._patch_embed_shape = patch_embed_shape
		self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
		self.num_temporal_patch = patch_embed_shape[0]

		if self.cls_embed_on:
			self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
			num_patches = self.num_spatial_patch * self.num_temporal_patch + 1
		else:
			num_patches = self.num_spatial_patch * self.num_temporal_patch

		if self.sep_pos_embed:
			self.pos_embed_spatial = nn.Parameter(
				torch.zeros(1, self.num_spatial_patch, embed_dim)
			)
			self.pos_embed_temporal = nn.Parameter(
				torch.zeros(1, self.num_temporal_patch, embed_dim)
			)
			if self.cls_embed_on:
				self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
		else:
			self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

	@property
	def patch_embed_shape(self):
		return self._patch_embed_shape

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x (torch.Tensor): Input tensor.
		"""
		B, N, C = x.shape
		if self.cls_embed_on:
			cls_tokens = self.cls_token.expand(B, -1, -1)
			x = torch.cat((cls_tokens, x), dim=1)

		if self.sep_pos_embed:
			pos_embed = self.pos_embed_spatial.repeat(
				1, self.num_temporal_patch, 1
			) + torch.repeat_interleave(
				self.pos_embed_temporal,
				self.num_spatial_patch,
				dim=1,
			)
			if self.cls_embed_on:
				pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
			x = x + pos_embed
		else:
			x = x + self.pos_embed

		return x

#### PATCH EMBEDDING ####

class PatchEmbed(nn.Module):
	'''
	Transformer basic patch embedding module. Performs patchifying input, flatten and
	and transpose.
	::
									   PatchModel
										   ↓
										flatten
										   ↓
									   transpose

	The builder can be found in `create_conv_patch_embed`.
	'''

	def __init__(
		self,
		*,
		patch_model: nn.Module = None,
	) -> None:
		super().__init__()
		set_attributes(self, locals())
		assert self.patch_model is not None

	def forward(self, x) -> torch.Tensor:
		x = self.patch_model(x)
		# B C (T) H W -> B (T)HW C
		return x.flatten(2).transpose(1, 2)


def create_conv_patch_embed(
	*,
	in_channels: int,
	out_channels: int,
	conv_kernel_size: Tuple[int] = (1, 16, 16),
	conv_stride: Tuple[int] = (1, 4, 4),
	conv_padding: Tuple[int] = (1, 7, 7),
	conv_bias: bool = True,
	conv: Callable = nn.Conv3d,
	) -> nn.Module:
	"""
	Creates the transformer basic patch embedding. It performs Convolution, flatten and
	transpose.
	::
										Conv3d
										   ↓
										flatten
										   ↓
									   transpose
	Args:
		in_channels (int): input channel size of the convolution.
		out_channels (int): output channel size of the convolution.
		conv_kernel_size (tuple): convolutional kernel size(s).
		conv_stride (tuple): convolutional stride size(s).
		conv_padding (tuple): convolutional padding size(s).
		conv_bias (bool): convolutional bias. If true, adds a learnable bias to the
			output.
		conv (callable): Callable used to build the convolution layer.
	Returns:
		(nn.Module): transformer patch embedding layer.
	"""
	conv_module = conv(
		in_channels=in_channels,
		out_channels=out_channels,
		kernel_size=conv_kernel_size,
		stride=conv_stride,
		padding=conv_padding,
		bias=conv_bias,
		
		)
	return PatchEmbed(patch_model=conv_module)

#### ATTENTION BLOCKS ####


class Mlp(nn.Module):
	"""
	A MLP block that contains two linear layers with a normalization layer. The MLP
	block is used in a transformer model after the attention block.

	::

						 Linear (in_features, hidden_features)
										   ↓
								 Normalization (act_layer)
										   ↓
								Dropout (p=dropout_rate)
										   ↓
						 Linear (hidden_features, out_features)
										   ↓
								Dropout (p=dropout_rate)
	"""

	def __init__( self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
		act_layer: Callable = nn.GELU, dropout_rate: float = 0.0, ) -> None:
		"""
		Args:
			in_features (int): Input feature dimension.
			hidden_features (Optional[int]): Hidden feature dimension. By default,
				hidden feature is set to input feature dimension.
			out_features (Optional[int]): Output feature dimension. By default, output
				features dimension is set to input feature dimension.
			act_layer (Callable): Activation layer used after the first linear layer.
			dropout_rate (float): Dropout rate after each linear layer. Dropout is not used
				by default.
		"""
		super().__init__()
		self.dropout_rate = dropout_rate
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		if self.dropout_rate > 0.0:
			self.dropout = nn.Dropout(dropout_rate)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x (tensor): Input tensor.
		"""
		x = self.fc1(x)
		x = self.act(x)
		if self.dropout_rate > 0.0:
			x = self.dropout(x)
		x = self.fc2(x)
		if self.dropout_rate > 0.0:
			x = self.dropout(x)
		return x


def _attention_pool(tensor: torch.Tensor, pool: Optional[Callable], thw_shape: List[int], has_cls_embed: bool = True, norm: Optional[Callable] = None,) -> torch.Tensor:
	'''
	Apply pool to a flattened input (given pool operation and the unflattened shape).


										 Input
										   ↓
										Reshape
										   ↓
										  Pool
										   ↓
										Reshape
										   ↓
										  Norm


	Args:
		tensor (torch.Tensor): Input tensor.
		pool (Optional[Callable]): Pool operation that is applied to the input tensor.
			If pool is none, return the input tensor.
		thw_shape (List): The shape of the input tensor (before flattening).
		has_cls_embed (bool): Whether the input tensor contains cls token. Pool
			operation excludes cls token.
		norm: (Optional[Callable]): Optional normalization operation applied to tensor
			after pool.

	Returns:
		tensor (torch.Tensor): Input tensor after pool.
		thw_shape (List[int]): Output tensor shape (before flattening).
	'''

	if pool is None:
		return tensor, thw_shape
	tensor_dim = tensor.ndim
	if tensor_dim == 4:
		pass
	elif tensor_dim == 3:
		tensor = tensor.unsqueeze(1)
	else:
		raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

	if has_cls_embed:
		cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

	B, N, L, C = tensor.shape
	T, H, W = thw_shape
	tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

	tensor = pool(tensor)

	thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
	L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
	tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
	if has_cls_embed:
		tensor = torch.cat((cls_tok, tensor), dim=2)
	if norm is not None:
		tensor = norm(tensor)

	if tensor_dim == 4:
		pass
	else:  # For the case tensor_dim == 3.
		tensor = tensor.squeeze(1)
	return tensor, thw_shape


class MultiScaleAttention(nn.Module):
	"""
	Implementation of a multiscale attention block. Compare to a conventional attention
	block, a multiscale attention block optionally supports pooling (either
	before or after qkv projection). If pooling is not used, a multiscale attention
	block is equivalent to a conventional attention block.

	::
								   Input
									 |
					|----------------|-----------------|
					↓                ↓                 ↓
				  Linear           Linear            Linear
					&                &                 &
				 Pool (Q)         Pool (K)          Pool (V)
					→ -------------- ←                 |
							 ↓                         |
					   MatMul & Scale                  |
							 ↓                         |
						  Softmax                      |
							 → ----------------------- ←
										 ↓
								   MatMul & Scale
										 ↓
									  DropOut
	"""

	def __init__( self, dim: int, num_heads: int = 8, qkv_bias: bool = False, dropout_rate: float = 0.0,
		kernel_q: _size_3_t = (1, 1, 1), kernel_kv: _size_3_t = (1, 1, 1), stride_q: _size_3_t = (1, 1, 1),
		stride_kv: _size_3_t = (1, 1, 1), norm_layer: Callable = nn.LayerNorm, has_cls_embed: bool = True, 
		pool_mode: str = "conv", pool_first: bool = False, ) -> None:

		"""
		Args:
			dim (int): Input feature dimension.
			num_heads (int): Number of heads in the attention layer.
			qkv_bias (bool): If set to False, the qkv layer will not learn an additive
				bias. Default: False.
			dropout_rate (float): Dropout rate.
			kernel_q (_size_3_t): Pooling kernel size for q. If both pooling kernel
				size and pooling stride size are 1 for all the dimensions, pooling is
				disabled.
			kernel_kv (_size_3_t): Pooling kernel size for kv. If both pooling kernel
				size and pooling stride size are 1 for all the dimensions, pooling is
				disabled.
			stride_q (_size_3_t): Pooling kernel stride for q.
			stride_kv (_size_3_t): Pooling kernel stride for kv.
			norm_layer (nn.Module): Normalization layer used after pooling.
			has_cls_embed (bool): If set to True, the first token of the input tensor
				should be a cls token. Otherwise, the input tensor does not contain a
				cls token. Pooling is not applied to the cls token.
			pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
				(average pooling), and "max" (max pooling).
			pool_first (bool): If set to True, pool is applied before qkv projection.
				Otherwise, pool is applied after qkv projection. Default: False.
		"""

		super().__init__()
		assert pool_mode in ["conv", "avg", "max"]

		self.pool_first = pool_first
		self.dropout_rate = dropout_rate
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5
		self.has_cls_embed = has_cls_embed
		padding_q = [int(q // 2) for q in kernel_q]
		padding_kv = [int(kv // 2) for kv in kernel_kv]

		self.q = nn.Linear(dim, dim, bias=qkv_bias)
		self.k = nn.Linear(dim, dim, bias=qkv_bias)
		self.v = nn.Linear(dim, dim, bias=qkv_bias)
		self.proj = nn.Linear(dim, dim)
		if dropout_rate > 0.0:
			self.proj_drop = nn.Dropout(dropout_rate)

		# Skip pooling with kernel and stride size of (1, 1, 1).
		if (
			kernel_q is not None
			and numpy.prod(kernel_q) == 1
			and numpy.prod(stride_q) == 1
		):
			kernel_q = None
		if (
			kernel_kv is not None
			and numpy.prod(kernel_kv) == 1
			and numpy.prod(stride_kv) == 1
		):
			kernel_kv = None

		if pool_mode in ("avg", "max"):
			pool_op = nn.MaxPool3d if pool_mode == "max" else nn.AvgPool3d
			self.pool_q = (
				pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
				if kernel_q is not None
				else None
			)
			self.pool_k = (
				pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
				if kernel_kv is not None
				else None
			)
			self.pool_v = (
				pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
				if kernel_kv is not None
				else None
			)
		elif pool_mode == "conv":
			self.pool_q = (
				nn.Conv3d(
					head_dim,
					head_dim,
					kernel_q,
					stride=stride_q,
					padding=padding_q,
					groups=head_dim,
					bias=False,
				)
				if kernel_q is not None
				else None
			)
			self.norm_q = norm_layer(head_dim) if kernel_q is not None else None
			self.pool_k = (
				nn.Conv3d(
					head_dim,
					head_dim,
					kernel_kv,
					stride=stride_kv,
					padding=padding_kv,
					groups=head_dim,
					bias=False,
				)
				if kernel_kv is not None
				else None
			)
			self.norm_k = norm_layer(head_dim) if kernel_kv is not None else None
			self.pool_v = (
				nn.Conv3d(
					head_dim,
					head_dim,
					kernel_kv,
					stride=stride_kv,
					padding=padding_kv,
					groups=head_dim,
					bias=False,
				)
				if kernel_kv is not None
				else None
			)
			self.norm_v = norm_layer(head_dim) if kernel_kv is not None else None
		else:
			raise NotImplementedError(f"Unsupported model {pool_mode}")

	def _qkv_proj(
		self,
		q: torch.Tensor,
		q_size: List[int],
		k: torch.Tensor,
		k_size: List[int],
		v: torch.Tensor,
		v_size: List[int],
		batch_size: List[int],
		chan_size: List[int],
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		q = (
			self.q(q)
			.reshape(batch_size, q_size, self.num_heads, chan_size // self.num_heads)
			.permute(0, 2, 1, 3)
		)
		k = (
			self.k(k)
			.reshape(batch_size, k_size, self.num_heads, chan_size // self.num_heads)
			.permute(0, 2, 1, 3)
		)
		v = (
			self.v(v)
			.reshape(batch_size, v_size, self.num_heads, chan_size // self.num_heads)
			.permute(0, 2, 1, 3)
		)
		return q, k, v

	def _qkv_pool(
		self,
		q: torch.Tensor,
		k: torch.Tensor,
		v: torch.Tensor,
		thw_shape: Tuple[torch.Tensor, List[int]],
	) -> Tuple[
		torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]
	]:
		q, q_shape = _attention_pool(
			q,
			self.pool_q,
			thw_shape,
			has_cls_embed=self.has_cls_embed,
			norm=self.norm_q if hasattr(self, "norm_q") else None,
		)
		k, k_shape = _attention_pool(
			k,
			self.pool_k,
			thw_shape,
			has_cls_embed=self.has_cls_embed,
			norm=self.norm_k if hasattr(self, "norm_k") else None,
		)
		v, v_shape = _attention_pool(
			v,
			self.pool_v,
			thw_shape,
			has_cls_embed=self.has_cls_embed,
			norm=self.norm_v if hasattr(self, "norm_v") else None,
		)
		return q, q_shape, k, k_shape, v, v_shape

	def _get_qkv_length(
		self,
		q_shape: List[int],
		k_shape: List[int],
		v_shape: List[int],
	) -> Tuple[int]:
		q_N = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
		k_N = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
		v_N = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)
		return q_N, k_N, v_N

	def _reshape_qkv_to_seq(
		self,
		q: torch.Tensor,
		k: torch.Tensor,
		v: torch.Tensor,
		q_N: int,
		v_N: int,
		k_N: int,
		B: int,
		C: int,
	) -> Tuple[int]:
		q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
		v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
		k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
		return q, k, v

	def forward(self, x: torch.Tensor, thw_shape: List[int], return_attentions: bool=False) -> Tuple[torch.Tensor, List[int]]:
		"""
		Args:
			x (torch.Tensor): Input tensor.
			thw_shape (List): The shape of the input tensor (before flattening).
		"""

		B, N, C = x.shape
		if self.pool_first:
			x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
			q = k = v = x
			q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)
			q_N, k_N, v_N = self._get_qkv_length(q_shape, k_shape, v_shape)
			q, k, v = self._reshape_qkv_to_seq(q, k, v, q_N, v_N, k_N, B, C)
			q, k, v = self._qkv_proj(q, q_N, k, k_N, v, v_N, B, C)
		else:
			q = k = v = x
			q, k, v = self._qkv_proj(q, N, k, N, v, N, B, C)
			q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)

		N = q.shape[2]
		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		if self.dropout_rate > 0.0:
			x = self.proj_drop(x)

		if return_attentions == True:
			return x, q_shape, attn
		else:
			return x, q_shape


class MultiScaleBlock(nn.Module):
	"""
	Implementation of a multiscale vision transformer block. Each block contains a
	multiscale attention layer and a Mlp layer.

	::


									  Input
										|-------------------+
										↓                   |
									   Norm                 |
										↓                   |
								MultiScaleAttention        Pool
										↓                   |
									 DropPath               |
										↓                   |
									Summation ←-------------+
										|
										|-------------------+
										↓                   |
									   Norm                 |
										↓                   |
									   Mlp                 Proj
										↓                   |
									 DropPath               |
										↓                   |
									Summation  ←------------+
	"""

	def __init__( self, dim: int, dim_out: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False,
	dropout_rate: float = 0.0, droppath_rate: float = 0.0, act_layer: nn.Module = nn.GELU, norm_layer: nn.Module =
	nn.LayerNorm, kernel_q: _size_3_t = (1, 1, 1), kernel_kv: _size_3_t = (1, 1, 1), stride_q: _size_3_t = (1, 1, 1),
	stride_kv: _size_3_t = (1, 1, 1), pool_mode: str = "conv", has_cls_embed: bool = True, pool_first: bool =
	False, ) -> None:
		"""
		Args:
			dim (int): Input feature dimension.
			dim_out (int): Output feature dimension.
			num_heads (int): Number of heads in the attention layer.
			mlp_ratio (float): Mlp ratio which controls the feature dimension in the
				hidden layer of the Mlp block.
			qkv_bias (bool): If set to False, the qkv layer will not learn an additive
				bias. Default: False.
			dropout_rate (float): DropOut rate. If set to 0, DropOut is disabled.
			droppath_rate (float): DropPath rate. If set to 0, DropPath is disabled.
			act_layer (nn.Module): Activation layer used in the Mlp layer.
			norm_layer (nn.Module): Normalization layer.
			kernel_q (_size_3_t): Pooling kernel size for q. If pooling kernel size is
				1 for all the dimensions, pooling is not used (by default).
			kernel_kv (_size_3_t): Pooling kernel size for kv. If pooling kernel size
				is 1 for all the dimensions, pooling is not used. By default, pooling
				is disabled.
			stride_q (_size_3_t): Pooling kernel stride for q.
			stride_kv (_size_3_t): Pooling kernel stride for kv.
			pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
				(average pooling), and "max" (max pooling).
			has_cls_embed (bool): If set to True, the first token of the input tensor
				should be a cls token. Otherwise, the input tensor does not contain a
				cls token. Pooling is not applied to the cls token.
			pool_first (bool): If set to True, pool is applied before qkv projection.
				Otherwise, pool is applied after qkv projection. Default: False.
		"""
		super().__init__()
		self.dim = dim
		self.dim_out = dim_out
		self.norm1 = norm_layer(dim)
		kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
		stride_skip = stride_q
		padding_skip = [int(skip // 2) for skip in kernel_skip]
		self.attn = MultiScaleAttention(
			dim,
			num_heads=num_heads,
			qkv_bias=qkv_bias,
			dropout_rate=dropout_rate,
			kernel_q=kernel_q,
			kernel_kv=kernel_kv,
			stride_q=stride_q,
			stride_kv=stride_kv,
			norm_layer=nn.LayerNorm,
			has_cls_embed=has_cls_embed,
			pool_mode=pool_mode,
			pool_first=pool_first,
		)
		self.drop_path = (
			DropPath(droppath_rate) if droppath_rate > 0.0 else nn.Identity()
		)
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.has_cls_embed = has_cls_embed
		self.mlp = Mlp(
			in_features=dim,
			hidden_features=mlp_hidden_dim,
			out_features=dim_out,
			act_layer=act_layer,
			dropout_rate=dropout_rate,
		)
		if dim != dim_out:
			self.proj = nn.Linear(dim, dim_out)

		self.pool_skip = (
			nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
			if len(kernel_skip) > 0
			else None
		)

	def forward(
		self, x: torch.Tensor, thw_shape: List[int], return_attentions: bool=False) -> Tuple[torch.Tensor, List[int]]:
		"""
		Args:
			x (torch.Tensor): Input tensor.
			thw_shape (List): The shape of the input tensor (before flattening).
		"""
		if return_attentions == True:
			x_block, thw_shape_new, attn = self.attn(self.norm1(x), thw_shape, return_attentions)
		else:
			x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape, return_attentions)

		x_res, _ = _attention_pool(
			x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
		)
		x = x_res + self.drop_path(x_block)
		x_norm = self.norm2(x)
		x_mlp = self.mlp(x_norm)
		if self.dim != self.dim_out:
			x = self.proj(x_norm)
		x = x + self.drop_path(x_mlp)
		if return_attentions == True:
			return x, thw_shape_new, attn
		else:
			return x, thw_shape_new

class SequencePool(nn.Module):
	"""
	Sequence pool produces a single embedding from a sequence of embeddings. Currently
	it supports "mean" and "cls".
	"""

	def __init__(self, mode: str) -> None:
		"""
		Args:
			mode (str): Optionals include "cls" and "mean". If set to "cls", it assumes
				the first element in the input is the cls token and returns it. If set
				to "mean", it returns the mean of the entire sequence.
		"""
		super().__init__()
		assert mode in ["cls", "mean"], "Unsupported mode for SequencePool."
		self.mode = mode

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.mode == "cls":
			x = x[:, 0]
		elif self.mode == "mean":
			x = x.mean(1)
		else:
			raise NotImplementedError
		return x


class VisionTransformerBasicHead(nn.Module):
	"""
	Vision transformer basic head.
	::
									  SequencePool
										   ↓
										Dropout
										   ↓
									   Projection
										   ↓
									   Activation
	The builder can be found in `create_vit_basic_head`.
	"""

	def __init__(
		self,
		sequence_pool: nn.Module = None,
		dropout: nn.Module = None,
		proj: nn.Module = None,
		activation: nn.Module = None,
	) -> None:
		"""
		Args:
			sequence_pool (torch.nn.modules): pooling module.
			dropout(torch.nn.modules): dropout module.
			proj (torch.nn.modules): project module.
			activation (torch.nn.modules): activation module.
		"""
		super().__init__()
		set_attributes(self, locals())
		assert self.proj is not None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Performs pooling.
		if self.sequence_pool is not None:
			x = self.sequence_pool(x)

		# Performs dropout.
		if self.dropout is not None:
			x = self.dropout(x)
		# Performs projection.
		x = self.proj(x)
		# Performs activation.
		if self.activation is not None:
			x = self.activation(x)
		return x


#### UTILS ####

def drop_path(
	x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
	) -> torch.Tensor:
	"""
	Stochastic Depth per sample.
	Args:
		x (tensor): Input tensor.
		drop_prob (float): Probability to apply drop path.
		training (bool): If True, apply drop path to input. Otherwise (tesing), return input.
	"""
	if drop_prob == 0.0 or not training:
		return x
	keep_prob = 1 - drop_prob
	shape = (x.shape[0],) + (1,) * (
		x.ndim - 1
	)  # work with diff dim tensors, not just 2D ConvNets
	mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
	mask.floor_()  # binarize
	output = x.div(keep_prob) * mask
	return output


class DropPath(nn.Module):
	"""
	Drop paths (Stochastic Depth) per sample.
	"""

	def __init__(self, drop_prob: float = 0.0) -> None:
		"""
		Args:
			drop_prob (float): Probability to apply drop path.
		"""
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x (tensor): Input tensor.
		"""
		return drop_path(x, self.drop_prob, self.training)
		
def set_attributes(self, params: List[object] = None) -> None:
	"""
	An utility function used in classes to set attributes from the input list of parameters.
	Args:
		params (list): list of parameters.
	"""
	if params:
		for k, v in params.items():
			if k != "self":
				setattr(self, k, v)


def round_width(width, multiplier, min_width=8, divisor=8, ceil=False):
	"""
	Round width of filters based on width multiplier
	Args:
		width (int): the channel dimensions of the input.
		multiplier (float): the multiplication factor.
		min_width (int): the minimum width after multiplication.
		divisor (int): the new width should be dividable by divisor.
		ceil (bool): If True, use ceiling as the rounding method.
	"""
	if not multiplier:
		return width

	width *= multiplier
	min_width = min_width or divisor
	if ceil:
		width_out = max(min_width, int(math.ceil(width / divisor)) * divisor)
	else:
		width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
	if width_out < 0.9 * width:
		width_out += divisor
	return int(width_out)


def round_repeats(repeats, multiplier):
	"""
	Round number of layers based on depth multiplier.
	"""
	if not multiplier:
		return repeats
	return int(math.ceil(multiplier * repeats))

def _init_vit_weights(model: nn.Module, trunc_normal_std: float = 0.02) -> None:
	"""
	Weight initialization for vision transformers.
	Args:
		model (nn.Module): Model to be initialized.
		trunc_normal_std (float): the expected standard deviation for fully-connected
			layer and ClsPositionalEncoding.
	"""
	for m in model.modules():
		if isinstance(m, nn.Linear):
			nn.init.trunc_normal_(m.weight, std=trunc_normal_std)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, SpatioTemporalClsPositionalEncoding):
			for weights in m.parameters():
				nn.init.trunc_normal_(weights, std=trunc_normal_std)


def init_net_weights(
	model: nn.Module,
	init_std: float = 0.01,
	style: str = "vit",
	) -> None:
	"""
	Performs weight initialization. Options include ResNet style weight initialization
	and transformer style weight initialization.
	Args:
		model (nn.Module): Model to be initialized.
		init_std (float): The expected standard deviation for initialization.
		style (str): Options include "resnet" and "vit".
	"""
	assert style in ["resnet", "vit"]
	if style == 'vit':
		return _init_vit_weights(model, init_std)
	else:
		raise NotImplementedError

#### CHECKPOINTS AND CONFIGS ####

checkpoint_paths = {
	"mvit_base_16x4": "{}/kinetics/MVIT_B_16x4.pyth".format(MODEL_ZOO_ROOT_DIR),
	"mvit_base_32x3": "{}/kinetics/MVIT_B_32x3_f294077834.pyth".format(
		MODEL_ZOO_ROOT_DIR
	),
	"mvit_base_16": "{}/imagenet/MVIT_B_16_f292487636.pyth".format(MODEL_ZOO_ROOT_DIR),
}

mvit_video_base_config = {
	"spatial_size": 224,
	"temporal_size": 16,
	"embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
	"atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
	"pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
	"pool_kv_stride_adaptive": [1, 8, 8],
	"pool_kvq_kernel": [3, 3, 3],
}

mvit_video_base_32x3_config = {
	"spatial_size": 224,
	"temporal_size": 32,
	"embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
	"atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
	"pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
	"pool_kv_stride_adaptive": [1, 8, 8],
	"pool_kvq_kernel": [3, 3, 3],
}

mvit_image_base_16_config = {
	"spatial_size": 224,
	"temporal_size": 1,
	"depth": 16,
	"conv_patch_embed_kernel": [7, 7],
	"conv_patch_embed_stride": [4, 4],
	"conv_patch_embed_padding": [3, 3],
	"use_2d_patch": True,
	"embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
	"atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
	"pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
	"pool_kv_stride_adaptive": [1, 4, 4],
	"pool_kvq_kernel": [1, 3, 3],
}


#### TORCH HUB BUILDER FUNCTIONS ####

def hub_model_builder(
	model_builder_func: Callable,
	pretrained: bool = False,
	progress: bool = True,
	checkpoint_path: str = "",
	default_config: Optional[Dict[Any, Any]] = None,
	**kwargs: Any,
) -> nn.Module:
	"""
	model_builder_func (Callable): Model builder function.
	pretrained (bool): Whether to load a pretrained model or not. Default: False.
	progress (bool): Whether or not to display a progress bar to stderr. Default: True.
	checkpoint_path (str): URL of the model weight to download.
	default_config (Dict): Default model configs that is passed to the model builder.
	**kwargs: (Any): Additional model configs. Do not modify the model configuration
	via the kwargs for pretrained model.
	"""
	if pretrained:
		assert len(kwargs) == 0, "Do not change kwargs for pretrained model."

	if default_config is not None:
		for argument, value in default_config.items():
			if kwargs.get(argument) is None:
				kwargs[argument] = value

	model = model_builder_func(**kwargs)
	if pretrained:
		# All models are loaded onto CPU by default
		checkpoint = load_state_dict_from_url(
			checkpoint_path, progress=progress, map_location="cpu"
		)
		state_dict = checkpoint["model_state"]
		model.load_state_dict(state_dict)
	return model

def mvit_base_16x4(
	pretrained: bool = False,
	progress: bool = True,
	**kwargs: Any,
	) -> nn.Module:
	"""
	Multiscale Vision Transformers model architecture [1] trained with an 16x4
	setting on the Kinetics400 dataset. Model with pretrained weights has top1
	accuracy of 78.9%.
	[1] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra
	Malik, Christoph Feichtenhofer, "Multiscale Vision Transformers"
	https://arxiv.org/pdf/2104.11227.pdf
	Args:
		pretrained (bool): If True, returns a model pre-trained on Kinetics400 dataset.
		progress (bool): If True, displays a progress bar of the download to stderr.
		kwargs: Use these to modify any of the other model settings. All the
			options are defined in create_multiscale_vision_transformers.
	NOTE: to use the pretrained model, do not modify the model configuration
	via the kwargs. Only modify settings via kwargs to initialize a new model
	without pretrained weights.
	"""

	return hub_model_builder(
		model_builder_func=create_multiscale_vision_transformers,
		pretrained=pretrained,
		progress=progress,
		checkpoint_path=checkpoint_paths["mvit_base_16x4"],
		default_config=mvit_video_base_config,
		**kwargs,
	)

def mvit_base_32x3(
	pretrained: bool = False,
	progress: bool = True,
	**kwargs: Any,
	) -> nn.Module:
	"""
	Multiscale Vision Transformers model architecture [1] trained with an 32x3
	setting on the Kinetics400 dataset. Model with pretrained weights has top1
	accuracy of 80.3%.
	[1] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra
	Malik, Christoph Feichtenhofer, "Multiscale Vision Transformers"
	https://arxiv.org/pdf/2104.11227.pdf
	Args:
		pretrained (bool): If True, returns a model pre-trained on Kinetics400 dataset.
		progress (bool): If True, displays a progress bar of the download to stderr.
		kwargs: Use these to modify any of the other model settings. All the
			options are defined in create_multiscale_vision_transformers.
	NOTE: to use the pretrained model, do not modify the model configuration
	via the kwargs. Only modify settings via kwargs to initialize a new model
	without pretrained weights.
	"""

	return hub_model_builder(
		model_builder_func=create_multiscale_vision_transformers,
		pretrained=pretrained,
		progress=progress,
		checkpoint_path=checkpoint_paths["mvit_base_32x3"],
		default_config=mvit_video_base_32x3_config,
		**kwargs,
	)

if __name__ == '__main__':

		'''
		DEBUG to build model and attempt to load weights 
		'''

		model = mvit_base_32x3(pretrained=True)
		print(model)


