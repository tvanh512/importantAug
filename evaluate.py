import torch
import os  
import numpy as np  
from tools.score_utils import get_likely_index,number_of_correct
import torch.nn.functional as F
from tools.data_utils import SubsetSC,path_index, label_to_index, index_to_label, get_same_padding, data_processing, musan_loader, load_config

def evaluate(args,
			criterion,
			data_loader,
			device,
			gen_flag,
			labels,
			model, 
			stft_transform,
			index_to_path=None,
			mask_add_aug=False,
			mask_forward_eval=False,
			mask_loss_eval= False,
			result_path="",
			save_mask=False,
			save_path=None,
			save_result=False,
			suffix=""):
	model.eval()
	correct,loss_epoch = 0,0
	wav_paths,targets,preds = [],[],[]
	with torch.no_grad():
		for data, target,wav_index in data_loader:
			data = data.to(device)
			target = target.to(device)
			#print('model.training',model.training)
			if not gen_flag:
				output = model(data)
			else:
				if mask_forward_eval:
					output,mask,x1,x2 = model(data,stft_transform,mask_forward_eval,mask_add_aug) 
					mask = torch.clip(mask,args.eps,1-args.eps) 
				else:
					output = model(data,stft_transform,mask_forward_eval,mask_add_aug) 
			pred = get_likely_index(output)
			correct += number_of_correct(pred, target)
			if mask_loss_eval:
				loss_mask_mean = args.weight_loss_mask_mean * torch.mean(mask)
				loss_classifier = args.weight_loss_classifier * criterion(output.squeeze(), target)
				loss_mask_entropy = args.weight_loss_mask_entropy * torch.mean(mask * torch.log(mask) + (1.0-mask) * torch.log(1.0-mask))
				loss_mask_variation = args.weight_loss_mask_freq_var * torch.mean(mask[:,:,1:,:]-mask[:,:,:-1,:]) + args.weight_loss_mask_time_var * torch.mean(mask[:,:,:,1:]-mask[:,:,:,:-1])
				loss = loss_classifier - loss_mask_mean - loss_mask_entropy + loss_mask_variation
			else:
				loss = criterion(output.squeeze(), target)
			loss_epoch += loss.item() * target.shape[0]
			if save_result:
				wav_paths += [index_to_path[i] for i in wav_index.cpu().detach().numpy()]  
				targets += [index_to_label(i,labels) for i in target.cpu().detach().numpy()] 
				preds+= [index_to_label(i,labels) for i in pred.squeeze().cpu().detach().numpy()] 
			
	#if mask_flag and save_mask:
	#    torch.save(mask,os.path.join(save_path,'mask_'+suffix +'.pt'))
	#    torch.save(data,os.path.join(save_path,'data_'+suffix +'.pt'))
	#    torch.save(x1,os.path.join(save_path,'x1_'+suffix +'.pt'))
	#    torch.save(x2,os.path.join(save_path,'x2_'+suffix +'.pt'))
	if save_result:
		np.savez(result_path, wav_paths=wav_paths, targets=targets,preds=preds)
	acc = 100. * correct / len(data_loader.dataset)
	avg_loss =  loss_epoch / len(data_loader.dataset)
	return acc,avg_loss

