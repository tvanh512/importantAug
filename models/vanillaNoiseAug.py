import torch
import torch.nn as nn
import tools
from tools.data_utils import get_same_padding
from models.generator import MaskGen_CNN
from models.load_model import load_classifier,load_generator
import torch
import torchaudio
from tools.audio_utils import make_noise,SNR_cal
import os

class vanillaNoiseAug(nn.Module):
	def __init__(self,args):
		super(vanillaNoiseAug, self).__init__()
		self.classifier,_ = load_classifier(args)
		self.wave_shape = args.wave_shape
		self.amptodB_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude')
		self.device = args.device
		self.args=args

	def load_best_classifier(self,path):
		cpkt = torch.load(path)
		self.classifier.load_state_dict(cpkt['model_state_dict'])

	def forward(self,x,mask_forward=False,mask_add_aug=False):
		x= torch.view_as_complex(x)
		if mask_forward:
			noise = make_noise(self.wave_shape,x.shape[0],None,self.args).to(self.device)
			scale = (torch.mean(x.abs()**2)/(torch.mean(noise.abs()**2) * 10.0**(self.args.target_SNR_dB/10.0)))**0.5
			x = x +  scale * noise
		out = self.classifier(x)
		return out

class vanillaNoiseAug2(nn.Module):
	def __init__(self,args):
		super(vanillaNoiseAug2, self).__init__()
		self.classifier,_ = load_classifier(args)
		self.wave_shape = args.wave_shape
		self.amptodB_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude')
		self.device = args.device
		self.args=args
		self.noise_1_weight = args.noise_1_weight
		self.noise_2_weight = args.noise_2_weight

	def load_best_classifier(self,path):
		cpkt = torch.load(path)
		self.classifier.load_state_dict(cpkt['model_state_dict'])

	def forward(self,x,mask_forward=False,mask_add_aug=False):
		x= torch.view_as_complex(x)
		if mask_forward:
			noise1 = self.noise_1_weight * make_noise(self.wave_shape,x.shape[0],None,self.args).to(self.device)
			noise2 = self.noise_2_weight * make_noise(self.wave_shape,x.shape[0],None,self.args).to(self.device)
			x = x +  noise1	
			x = x +  noise2
		out = self.classifier(x)
		return out