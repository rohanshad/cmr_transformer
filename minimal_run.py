'''
Minimal working example for mvit based cmr_core encoder
'''

import torch
import os 
import model_factory 
from huggingface_hub import hf_hub_download
from torchvision import transforms

# Cuda if able #
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load model and pre-trained weights from HF page
ckpt_path = hf_hub_download(
    repo_id="rohanshad/cmr_c0.1",
    filename="epoch=599-step=124200.ckpt",
    revision="main"
)

model = model_factory.Cardiac_MRI_Encoder('mvit', 50, 512, 1, None, 16, 'linear', True)
model = model_factory.load_contrastive_pretrained_weights(model, ckpt_path)
model.to(device)

# Video as [B, C, F, H, W] for model input
# Input dims: [B, 3, 16, 224, 224]

## If using transforms on raw dicom or hdf data:
# val_transforms = transforms.Compose([transforms.Resize(size=224+20),transforms.CenterCrop(size=224),])

# Demo input for show:
demo_input = torch.rand([1, 3, 16, 224, 224])
demo_input = demo_input.to(device)

print('------------------------------------')
print(f'Using device: {device}')
video_emb = model(demo_input)
print(video_emb)
print(f'video embedding shape: {video_emb.shape}')