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
import numpy as np

class impAug1(nn.Module):
	def __init__(self,args):
		super(impAug1, self).__init__()
		self.gen,_ = load_generator(args) 
		self.classifier,_ = load_classifier(args)
		self.mask_option = args.mask_option
		self.wave_shape = args.wave_shape
		self.amptodB_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude')
		self.noise_1_weight = args.noise_1_weight
		self.noise_2_weight = args.noise_2_weight
		self.noise_3_weight = args.noise_3_weight
		self.gen_augment = args.gen_augment
		self.device = args.device
		self.args=args
		self.mask_shift = args.mask_shift
		self.gen_aug_type = args.gen_aug_type

	def frozen(self,m):
		m.eval()
		for p in m.parameters():
			p.requires_grad = False

	def unfrozen(self,m):
		m.train()
		for p in m.parameters():
			p.requires_grad = True

	def load_best_classifier(self,path):
		cpkt = torch.load(path)
		self.classifier.load_state_dict(cpkt['model_state_dict'])

	def forward(self,x,stft_transform,mask_forward,mask_add_aug=False):
		#figures_path ='./Figures/'
		
		if mask_forward:
			#torch.save(x.detach().cpu(),os.path.join(figures_path,'data.pt'))
			x= torch.view_as_complex(x)
			if not self.gen_augment:
				xdB = self.amptodB_transform(x.abs())
				mask = self.gen(xdB)
			else:
				if self.gen_aug_type == 1:
					noise3 = self.noise_3_weight * make_noise(self.wave_shape,x.shape[0],stft_transform,self.args).to(self.device)
					x = x +  noise3
					xdB = self.amptodB_transform(x.abs())
					mask = self.gen(xdB)
				elif self.gen_aug_type == 3:
					xdB = self.amptodB_transform(x.abs())
					mask = self.gen(xdB)
					#torch.save(mask.detach().cpu(),os.path.join(figures_path,'mask.pt'))
					ax = np.random.choice([2,3])
					shift = np.random.randint(-self.mask_shift,self.mask_shift)
					mask = torch.roll(mask, shift,ax)
					#torch.save(mask.detach().cpu(),os.path.join(figures_path,'mask_roll.pt'))
				elif self.gen_aug_type == 4:
					mask = torch.zeros(x.shape).to(self.device)
				elif self.gen_aug_type == 5:
					xdB = self.amptodB_transform(x.abs())
					mask = self.gen(xdB)
					
					ax = np.random.choice([2,3])
					shift = np.random.randint(-self.mask_shift,self.mask_shift)
					strategy = np.random.choice([0,1])
					if strategy ==0:
						#torch.save(mask.detach().cpu(),os.path.join(figures_path,'mask.pt'))
						mask = torch.roll(mask, shift,ax)
						#torch.save(mask.detach().cpu(),os.path.join(figures_path,'mask_roll.pt'))
					else:
						mask = torch.ones(x.shape).to(self.device)
			if self.mask_option ==1:
				if mask_add_aug:
					mask = mask * torch.rand(mask.shape).to(self.device)
				noise1 = self.noise_1_weight * make_noise(self.wave_shape,x.shape[0],stft_transform,self.args).to(self.device)
				x1 = x +  noise1
				noise2 = self.noise_2_weight * make_noise(self.wave_shape,x.shape[0],stft_transform,self.args).to(self.device)
				x2 = x1 +  mask * noise2
	
			elif self.mask_option == 2:
				if mask_add_aug:
					mask = mask * torch.rand(mask.shape)
				x1 = x 
				x2 = x * mask + 1e-5
			elif self.mask_option == 3:
				noise = make_noise(self.wave_shape,x.shape[0],None,self.args).to(self.device)
				#torch.save(noise.detach().cpu(),os.path.join(figures_path,'noise.pt'))
				scale = (torch.mean(x.abs()**2)/(torch.mean(noise.abs()**2) * 10.0**(self.args.target_SNR_dB/10.0)))**0.5
				noise = noise * mask
				x1 = x
				x2 = x +  scale * noise		
				#torch.save(x2.detach().cpu(),os.path.join(figures_path,'x2.pt'))
			elif self.mask_option == 4:
				#***********************************************************************************************
				# # last
				# # 1, 10, 20 
				q = self.args.q
				num_q = np.percentile(mask.cpu().detach().numpy(), q, axis=[1, 2, 3])
				num_q = torch.from_numpy(np.array(num_q)).to(device=x.device, dtype=torch.float32)
				for i, mask_i in enumerate(mask):
					mask[i] = torch.where(mask_i>num_q[i], torch.ones_like(mask_i), torch.zeros_like(mask_i))
				# #************************************************************************************************
				noise = make_noise(self.wave_shape,x.shape[0],None,self.args).to(self.device)
				scale = (torch.mean(x.abs()**2)/(torch.mean(noise.abs()**2) * 10.0**(self.args.target_SNR_dB/10.0)))**0.5
				x1 = x
				x2 = x +  scale * noise	
			else:
				x1 = x
				x2 = x *(1 + mask)
			out = self.classifier(x2)

			return out,mask,x1,x2
		else:
			out = self.classifier(x)
			return out