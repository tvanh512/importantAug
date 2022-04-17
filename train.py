from evaluate import evaluate
import torch.optim as optim
import torch
from models.load_model import load_classifier  
import os
from tools.score_utils import get_likely_index,number_of_correct
from tools.data_utils import path_index
import torch.nn.functional as F
import time

def train_epoch(args,
				criterion,
				device,
				epoch,
				gen_flag,
				mask_add_aug,
				mask_forward_train,
				mask_loss_train,
				model,
				optimizer,
				stft_transform,
				train_loader,
				writer,
				writer_prefix):
	# gen_flag: True mean the network has generator
	# mask_forward_train: there is a mask in the forward path calculation
	# mask_loss_train: there are related-mask losses in the final loss
	# Train mode, important when using dropout, batchnorm, etc.
	start_time = time.time()
	model.train()  
	loss_train_epoch,correct = 0.0,0.0
	loss_mask_mean_epoch, loss_classifier_epoch, loss_mask_entropy_epoch, loss_mask_variation_epoch = 0,0,0,0
	for batch_idx, (data, target) in enumerate(train_loader):
		data,target = data.to(device),target.to(device)
		
		# Zero the model parameter gradients for every batch
		optimizer.zero_grad()

		# If the network contains only classifier without mask generator
		if not gen_flag:
			if not args.vanillaNoiseAug:
				output = model(data)
			else:
				output = model(data,mask_forward_train,mask_add_aug)
			loss = criterion(output.squeeze(), target)
		# If the network has mask generator
		else:
			output,mask,x1,x2= model(data,stft_transform,mask_forward_train,mask_add_aug)
			mask = torch.clip(mask,args.eps,1-args.eps)  # mask shape B x F x T # note clip only work with min eps = 1e-5, 1e-6 will not work
			if mask_loss_train:
				loss_classifier = args.weight_loss_classifier * criterion(output.squeeze(), target)
				loss_mask_entropy = args.weight_loss_mask_entropy * torch.mean(torch.log(mask))
				loss_mask_variation = args.weight_loss_mask_freq_var * torch.mean(torch.abs(mask[:,:,1:,:]-mask[:,:,:-1,:])) + args.weight_loss_mask_time_var * torch.mean(torch.abs(mask[:,:,:,1:]-mask[:,:,:,:-1]))
				loss = loss_classifier - loss_mask_entropy + loss_mask_variation
			else:
				loss = criterion(output.squeeze(), target)
		
		# Compute gradient: dLoss/dparams
		loss.backward()

		# Update parameters: params = params - lr * dloss/dparams
		optimizer.step()

		pred = get_likely_index(output)
		correct += number_of_correct(pred, target)
		loss_train_epoch += loss.item() * target.shape[0]
		if mask_loss_train:
			loss_classifier_epoch += loss_classifier.item() * target.shape[0]
			loss_mask_entropy_epoch += loss_mask_entropy.item() * target.shape[0]
			loss_mask_variation_epoch += loss_mask_variation.item() * target.shape[0]
	acc_train = 100. * correct / len(train_loader.dataset)
	loss_train_avg =  loss_train_epoch / len(train_loader.dataset)
	writer.add_scalar(writer_prefix+'/acc_train', acc_train, epoch)	
	writer.add_scalar(writer_prefix+'/loss_train_avg',loss_train_avg, epoch)
	print("Epoch %d: Training time %.2f, Train acc %.2f, train loss  %.2f "%(epoch,time.time()- start_time,acc_train,loss_train_avg))
	if mask_loss_train:	
		loss_classifier_avg = loss_classifier_epoch / len(train_loader.dataset)
		loss_mask_entropy_avg = loss_mask_entropy_epoch / len(train_loader.dataset)
		loss_mask_variation_avg = loss_mask_variation_epoch / len(train_loader.dataset)	
		writer.add_scalar(writer_prefix+'/loss_classifier_avg', loss_classifier_avg, epoch)	
		writer.add_scalar(writer_prefix+'/loss_mask_entropy_avg',loss_mask_entropy_avg, epoch)
		writer.add_scalar(writer_prefix+'/loss_mask_variation_avg',loss_mask_variation_avg, epoch)
		print("Epoch %d: loss_classifier_avg %.2f,loss_mask_entropy_avg  %.2f,loss_mask_variation_avg %.2f "%(epoch,loss_classifier_avg,loss_mask_entropy_avg,loss_mask_variation_avg))
	if not gen_flag:
		return acc_train,loss_train_avg
	else:
		return acc_train,loss_train_avg,data,output,mask,x1,x2
		
def train(args,
		best_model_name,
		criterion,
		device,
		dev_loader,
		gen_flag, 
		labels,
		mask_add_aug,
		mask_forward_train,
		mask_loss_train,
		mask_forward_eval,
		mask_loss_eval,
		model,
		optimizer,
		scheduler,
		save_path,
		stft_transform,
		train_loader,
		writer,
		writer_prefix):
	# gen_flag: True mean the network has generator
	# mask_forward_train: there is a mask in the forward path calculation
	# mask_loss_train: there are related-mask losses in the final loss

	loss_dev_best= float("inf")
	for epoch in range(1, args.n_epoch + 1):
		# Training
		if not gen_flag:
			acc_train,loss_mean_train = train_epoch(args,
													criterion,
													device,
													epoch,
													gen_flag,
													mask_add_aug,
													mask_forward_train,
													mask_loss_train,
													model,
													optimizer,
													stft_transform,
													train_loader,
													writer,
													writer_prefix)
		else:
			acc_train,loss_mean_train,data,output,mask,x1,x2 = train_epoch(args,
																			criterion,
																			device,
																			epoch,
																			gen_flag,
																			mask_add_aug,
																			mask_forward_train,
																			mask_loss_train,
																			model,
																			optimizer,
																			stft_transform,
																			train_loader,
																			writer,
																			writer_prefix)
		# Doing inference on the development set						   
		acc_dev,loss_dev = evaluate(args,
									criterion,
									dev_loader,
									device,
									gen_flag,
									labels,
									model,
									stft_transform,
									save_path=save_path,
									mask_add_aug = False,
									mask_forward_eval=mask_forward_eval,
									mask_loss_eval = mask_loss_eval)
		print("Epoch %d: dev acc %.2f, dev loss  %.3f "%(epoch,acc_dev,loss_dev))
		
		# Learning rate schedule
		scheduler.step()

		# Early stopping and save the best model
		if loss_dev < loss_dev_best:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				},os.path.join(save_path,best_model_name))
			epoch_best,loss_dev_best,acc_dev_best = epoch,loss_dev,acc_dev
			if gen_flag:
				torch.save(mask,os.path.join(save_path,'mask.pt'))
				torch.save(output,os.path.join(save_path,'output.pt'))
				torch.save(data,os.path.join(save_path,'data.pt'))
				torch.save(x1,os.path.join(save_path,'x1.pt'))
				torch.save(x2,os.path.join(save_path,'x2.pt'))
		if epoch - epoch_best > args.patience:
			break	
	print(" Best Epoch %d: best dev acc %.2f, best dev loss  %.2f "%(epoch_best,acc_dev_best,loss_dev_best))
