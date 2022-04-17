import argparse
import numpy as np
import torch
from torch.functional import stft
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as aT
import torch.optim as optim
import torchaudio
from torch.utils.tensorboard import SummaryWriter
import glob
import random
from torch.utils.data import TensorDataset, DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import os
import scipy.stats as st
from tools.data_utils import SubsetSC,path_index, label_to_index, index_to_label, get_same_padding, data_processing, load_config,step_counter
from train import train
from tools.score_utils import confidence_interval
from models.gen_asr import impAug1
from models.load_model import load_classifier,load_generator 
import warnings 
from evaluate import evaluate
from tools.optimizer_utils import optimizer_helper
from evaluate_noisy import evaluate_noisy
warnings.filterwarnings("ignore")

def main(args):

	# Training on GPU or CPU
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print('Device: ',device)
	args.device = device
	kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if use_cuda else {}

	# Google command train/dev/test set
	speech_command_path= os.path.join(args.data_path,'SpeechCommands/speech_commands_v0.02')
	args.speech_command_path = speech_command_path
	train_set = SubsetSC(args.data_path,"training")
	dev_set = SubsetSC(args.data_path,"validation")
	test_set = SubsetSC(args.data_path,"testing")
	path_to_index_dev,index_to_path_dev = path_index(dev_set)
	path_to_index_test,index_to_path_test = path_index(test_set)
	labels =['backward','bed','bird', 'cat','dog', 'down', 'eight','five','follow',
	'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
	'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
	'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
	args.n_class = len(labels)

	n_feat = args.n_fft //2 + 1
	args.n_feat = n_feat
	stft_transform = torchaudio.transforms.Spectrogram(n_fft = args.n_fft,hop_length = args.hop_length,power = None)

	# Define data loader and paths
	train_loader = torch.utils.data.DataLoader(dataset=train_set,
							batch_size=args.batch_size,
							shuffle=True,
							collate_fn=lambda x: data_processing(x,'train',labels,stft_transform,args,None),
							**kwargs)
	dev_loader = torch.utils.data.DataLoader(dataset=dev_set,
							batch_size=args.batch_size,
							shuffle=False,
							collate_fn=lambda x: data_processing(x,'dev',labels,stft_transform,args,path_to_index_dev),
							**kwargs)
	test_loader = torch.utils.data.DataLoader(dataset=test_set,
							batch_size=args.batch_size,
							shuffle=False,
							collate_fn=lambda x: data_processing(x,'test',labels,stft_transform,args,path_to_index_test),
							**kwargs)

	dev_paths = os.path.join(args.speech_command_path,"validation_list.txt")
	test_paths = os.path.join(args.speech_command_path,"testing_list.txt")
	waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
	wave_shape = waveform.shape
	args.wave_shape = wave_shape
	criterion = torch.nn.NLLLoss()

	#--------------------------------------------------------------
	# Train classifier only
	# Classifier,optimization and loss set up

	if not args.save_path_clf:
		save_path_clf = os.path.join(args.save_path,args.exp_name,'classifier') 
	else:
		save_path_clf = args.save_path_clf
	print('save_path_clf',save_path_clf)
	if not os.path.isdir(save_path_clf):
		os.makedirs(save_path_clf)
	if not args.no_stage1:
		clf,args = load_classifier(args)
		print('Classifier',clf)
		if args.load_best_classifier_stage1:
			cpkt = torch.load(os.path.join(args.pre_train_clf,args.best_classier_name))
			clf.load_state_dict(cpkt['model_state_dict'])
		clf.to(device)
		clf_optimizer, clf_scheduler = optimizer_helper(clf,args)
		# Train a network with only classifier that classifies 35 classes
		# Set up Tensorboard
		writer_clf = SummaryWriter(log_dir=save_path_clf)
		clf_parameters = sum([param.nelement() for param in clf.parameters()])
		print('Number of classifier parameters',clf_parameters)
		train(args=args,
			best_model_name=args.best_classier_name,
			criterion=criterion,
			device=device,
			dev_loader=dev_loader,
			gen_flag=False,
			labels=labels,
			mask_add_aug=False,
			mask_forward_train=False,
			mask_loss_train=False,
			mask_forward_eval=False,
			mask_loss_eval=False,
			model=clf,
			optimizer=clf_optimizer,
			scheduler=clf_scheduler,
			save_path=save_path_clf,
			stft_transform=stft_transform,
			train_loader=train_loader,
			writer=writer_clf,
			writer_prefix='clf')

		# Load the best model and test on the test set
	if args.stage1_test:
		clf,args = load_classifier(args)
		cpkt = torch.load(os.path.join(save_path_clf,args.best_classier_name))
		clf.load_state_dict(cpkt['model_state_dict'])
		clf.to(device)
		acc_test,loss_test = evaluate(args=args,
										criterion=criterion,
										data_loader=test_loader,
										device=device,
										gen_flag=False,
										labels=labels,
										model=clf,
										stft_transform=stft_transform,
										save_path=save_path_clf,
										mask_add_aug=False,
										mask_forward_eval=False,
										mask_loss_eval=False)
		print("Test acc %.2f, test err %.2f, test loss  %.3f "%(acc_test,100-acc_test,loss_test))
	if args.stage1_test_musan:
		clf,args = load_classifier(args)
		cpkt = torch.load(os.path.join(save_path_clf,args.best_classier_name))
		clf.load_state_dict(cpkt['model_state_dict'])
		clf.to(device)
		evaluate_noisy(args=args,
					    criterion=criterion,
                        device=device,
                        gen_flag=False,
                        labels=labels,
                        model=clf,
                        stft_transform=stft_transform,
                        mask_add_aug=False,
                        mask_forward_eval=False,
                        mask_loss_eval=False,
                        save_path=None,
                        noisy_folder_name_base=args.musan_folder_name_base,
                        noisy_sc_path=args.musan_sc_path)

	if args.stage1_test_qut:
		clf,args = load_classifier(args)
		cpkt = torch.load(os.path.join(save_path_clf,args.best_classier_name))
		clf.load_state_dict(cpkt['model_state_dict'])
		clf.to(device)
		evaluate_noisy(args=args,
					    criterion=criterion,
                        device=device,
                        gen_flag=False,
                        labels=labels,
                        model=clf,
                        stft_transform=stft_transform,
                        mask_add_aug=False,
                        mask_forward_eval=False,
                        mask_loss_eval=False,
                        save_path=None,
                        noisy_folder_name_base=args.qut_folder_name_base,
                        noisy_sc_path=args.qut_sc_path)
		
	#--------------------------------------------------------------
	# 2. Train a network with generator and classifier.
	# The classifier parameters are frozen, only mask generator is trained
	# Initialize augment_model which has classifier and mask generator
	if not args.save_path_clf_frozen: 
		save_path_clf_frozen = os.path.join(args.save_path,args.exp_name,'clf_frozen') 
	else:
		save_path_clf_frozen = args.save_path_clf_frozen
	net = impAug1(args)
	net.to(device)
	if not args.no_stage2:
		
		# Load the best classifier model
		net.load_best_classifier(os.path.join(save_path_clf,args.best_classier_name))
		
		# Frozen the classifier
		net.frozen(net.classifier)

		# Check if frozen correct
		for p in net.gen.parameters():
			print('gen requires_grad: ',p.requires_grad)
		for p in net.classifier.parameters():
			print('classifier requires_grad: ',p.requires_grad)
		gen_parameters = sum([param.nelement() for param in net.gen.parameters()])
		print('Number of generator parameters',gen_parameters)
		net_clf_parameters = sum([param.nelement() for param in net.classifier.parameters()])
		print('Number of net classifier parameters',net_clf_parameters)

		#Optimization and loss set up
		gen_optimizer, gen_scheduler = optimizer_helper(net,args)
		mask_flag_net = True

		print('save_path_clf_frozen',save_path_clf_frozen)
		if not os.path.isdir(save_path_clf_frozen):
			os.makedirs(save_path_clf_frozen)
		writer_clf_frozen = SummaryWriter(log_dir= save_path_clf_frozen)
		
		train(args=args,
			best_model_name=args.best_clf_frozen_name,
			criterion=criterion,
			device=device,
			dev_loader=dev_loader,
			gen_flag=True,
			labels=labels,
			mask_add_aug=False,
			mask_forward_train=True,
			mask_loss_train=True,
			mask_forward_eval=True,
			mask_loss_eval=True,
			model=net,
			optimizer=gen_optimizer,
			scheduler=gen_scheduler,
			save_path=save_path_clf_frozen,
			stft_transform=stft_transform,
			train_loader=train_loader,
			writer=writer_clf_frozen,
			writer_prefix='clf_frozen')
		
	#--------------------------------------------------------------
	# 3. Train the above network with generator and classifier.
	# But now the generator parameters are frozen and classifier is trained
	# Load the best mask generator
	if not args.save_path_gen_frozen:
		save_path_gen_frozen = os.path.join(args.save_path,args.exp_name,'gen_frozen') 
	else:
		save_path_gen_frozen = args.save_path_gen_frozen
	net = impAug1(args)
	net.to(device)
	if not args.no_stage3:
		cpkt = torch.load(os.path.join(save_path_clf_frozen,args.best_clf_frozen_name))
		print('load best clf frozen model at ',os.path.join(save_path_clf_frozen,args.best_clf_frozen_name))
		net.load_state_dict(cpkt['model_state_dict'])
		
		# Frozen the generator
		net.frozen(net.gen)
		net.unfrozen(net.classifier)
		# Check if frozen correct
		for p in net.gen.parameters():
			print('gen requires_grad: ',p.requires_grad)
		for p in net.classifier.parameters():
			print('classifier requires_grad: ',p.requires_grad)

		print('save_path_gen_frozen',save_path_gen_frozen)
		if not os.path.isdir(save_path_gen_frozen):
			os.makedirs(save_path_gen_frozen)

		#Optimization and loss set up
		clf_optimizer_2, clf_scheduler_2 = optimizer_helper(net,args)
		writer_gen_frozen = SummaryWriter(log_dir=save_path_gen_frozen)
		
		train(args=args,
			best_model_name=args.best_gen_frozen_name,
			criterion=criterion,
			device=device,
			dev_loader=dev_loader,
			gen_flag=True,
			labels=labels,
			mask_add_aug=args.mask_add_aug,
			mask_forward_train=True,
			mask_loss_train=False,
			mask_forward_eval=False,
			mask_loss_eval=False,
			model=net,
			optimizer=clf_optimizer_2,
			scheduler=clf_scheduler_2,
			save_path=save_path_gen_frozen,
			stft_transform=stft_transform,
			train_loader=train_loader,
			writer=writer_gen_frozen,
			writer_prefix='gen_frozen')

	# Evaluate on standard Google Speech Command test set	
	if args.stage4:
		
		net = impAug1(args)
		net.to(device)
		cpkt = torch.load(os.path.join(save_path_gen_frozen,args.best_gen_frozen_name))
		print('load best gen frozen model at ',os.path.join(save_path_gen_frozen,args.best_gen_frozen_name))
		net.load_state_dict(cpkt['model_state_dict'])
		
		acc_test,loss_test = evaluate(args=args,
									criterion=criterion,
									data_loader=test_loader,
									device=device,
									gen_flag=True,
									labels=labels,
									model=net,
									stft_transform=stft_transform,
									mask_add_aug=False,
									mask_forward_eval=False,
									mask_loss_eval = False,
									save_path=None,
									)
		print("Test acc %.2f, test err %.2f, test loss  %.3f "%(acc_test,100-acc_test,loss_test))
	
	# Evaluate noisy dataset create by mixing Musan noisy and Google Speech Command
	if args.stage5:
		net = impAug1(args)
		net.to(device)
		cpkt = torch.load(os.path.join(save_path_gen_frozen,args.best_gen_frozen_name))
		print('load best gen frozen model at ',os.path.join(save_path_gen_frozen,args.best_gen_frozen_name))
		net.load_state_dict(cpkt['model_state_dict'])
		
		evaluate_noisy(args=args,
					    criterion=criterion,
                        device=device,
                        gen_flag=True,
                        labels=labels,
                        model=net,
                        stft_transform=stft_transform,
                        mask_add_aug=False,
                        mask_forward_eval=False,
                        mask_loss_eval=False,
                        save_path=None,
                        noisy_folder_name_base=args.musan_folder_name_base,
                        noisy_sc_path=args.musan_sc_path)

	# Evaluate noisy dataset create by mixing QUT noisy and Google Speech Command
	if args.stage6:
		net = impAug1(args)
		net.to(device)
		cpkt = torch.load(os.path.join(save_path_gen_frozen,args.best_gen_frozen_name))
		print('load best gen frozen model at ',os.path.join(save_path_gen_frozen,args.best_gen_frozen_name))
		net.load_state_dict(cpkt['model_state_dict'])
		
		evaluate_noisy(args=args,
					    criterion=criterion,
                        device=device,
                        gen_flag=True,
                        labels=labels,
                        model=net,
                        stft_transform=stft_transform,
                        mask_add_aug=False,
                        mask_forward_eval=False,
                        mask_loss_eval=False,
                        save_path=None,
                        noisy_folder_name_base=args.qut_folder_name_base,
                        noisy_sc_path=args.qut_sc_path)

	# Evaluate on standard Google Speech Command dev set	
	if args.stagedev:
		
		net = impAug1(args)
		net.to(device)
		cpkt = torch.load(os.path.join(save_path_gen_frozen,args.best_gen_frozen_name))
		print('load best gen frozen model at ',os.path.join(save_path_gen_frozen,args.best_gen_frozen_name))
		net.load_state_dict(cpkt['model_state_dict'])
		
		acc_dev,loss_dev = evaluate(args=args,
									criterion=criterion,
									data_loader=dev_loader,
									device=device,
									gen_flag=True,
									labels=labels,
									model=net,
									stft_transform=stft_transform,
									mask_add_aug=False,
									mask_forward_eval=False,
									mask_loss_eval = False,
									save_path=None,
									)
		print("Dev acc %.2f, dev err %.2f, dev loss  %.3f "%(acc_dev,100-acc_dev,loss_dev))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_name",default="config.cfg",type=str,help="config_name")
	parser.add_argument("--config_path",default="./config.cfg",type=str,help="Path to config file")
	parser.add_argument("--log_interval",type=int,default=300)
	parser.add_argument("--mel",default=False,action='store_true',help="Using Mel or linear STFT")
	parser.add_argument("--n_fft",type=int,default=512)
	parser.add_argument("--n_mel",type=int,default=80)
	parser.add_argument("--hop_length",type=int,default=128)
	parser.add_argument("--lr",type=float,default=0.001)
	parser.add_argument("--n_epoch",type=int,default=200)
	parser.add_argument("--patience",type=int,default=30)
	parser.add_argument("--weight_decay",type=float,default=0.0001)
	parser.add_argument("--sample_rate",type=int,default=16000)
	parser.add_argument("--batch_size",type=int,default=256)
	parser.add_argument("--kernel_size_cls",type=int,default=9)
	parser.add_argument("--kernel_size_gen",type=int,default=5)
	parser.add_argument("--out_channels_gen",type=int,default=1)
	parser.add_argument("--n_layer_cls",type=int,default=5)
	parser.add_argument("--n_layer_gen",type=int,default=2)
	parser.add_argument("--eps",type=float,default=0.0001)
	parser.add_argument("--weight_loss_mask_mean",type=float,default=1)
	parser.add_argument("--weight_loss_mask_entropy",type=float,default=1,help='lambda_e in equation 5 in the paper')
	parser.add_argument("--weight_loss_mask_freq_var",type=float,default=2,help='lambda_f in equation 5 in the paper')
	parser.add_argument("--weight_loss_mask_time_var",type=float,default=1,help='lambda_t in equation 5 in the paper')
	parser.add_argument("--weight_loss_classifier",type=float,default=3,help='weight of classifier loss, which is lambda_r in equation 5 in the paper')
	parser.add_argument("--noise_type",type=str,default='white')
	parser.add_argument("--noise_1_weight",type=float,default=0.001)
	parser.add_argument("--noise_2_weight",type=float,default=5.0)
	parser.add_argument("--noise_3_weight",type=float,default=1.0)
	parser.add_argument("--cosine_annealing",default=False,action='store_true')
	parser.add_argument("--mask_option",type=int,default=1)
	parser.add_argument("--num_workers",type=int,default=8)
	parser.add_argument("--seed",type=int,default=100)
	parser.add_argument("--num_exp",type=int,default=1)
	parser.add_argument("--data_path",type=str,default="./data")
	parser.add_argument("--save_path",type=str,default="./saved_model") 
	parser.add_argument("--save_path_clf",type=str,default=None)
	parser.add_argument("--save_path_clf_frozen",type=str,default=None)
	parser.add_argument("--save_path_gen_frozen",type=str,default=None)
	parser.add_argument("--exp_name",type=str,default="C1_clf_CNN_gen_CNN_1") 
	parser.add_argument("--get_test_result",default=False,action='store_true',help="Get result on test set")
	parser.add_argument("--dropout",type=float,default=0.0)
	parser.add_argument("--arch",type=str,default='Classifier_CNN')
	parser.add_argument("--arch_gen",type=str,default='MaskGen_CNN')
	parser.add_argument("--best_classier_name",type=str,default='best_classifier.pt')
	parser.add_argument("--best_clf_frozen_name",type=str,default='best_clf_frozen.pt')
	parser.add_argument("--best_gen_frozen_name",type=str,default='best_gen_frozen.pt')
	parser.add_argument("--no_stage1",default=False,action='store_true',help="Stage 1: train classifier")
	parser.add_argument("--stage1_test",default=False,action='store_true',help="Stage 1: test classifier")
	parser.add_argument("--stage1_test_musan",default=False,action='store_true',help="Stage 1: test classifier on musan")
	parser.add_argument("--stage1_test_qut",default=False,action='store_true',help="Stage 1: test classifier on QUT")
	parser.add_argument("--no_stage2",default=False,action='store_true',help="Stage 2: frozen classifier, train mask generator")
	parser.add_argument("--no_stage3",default=False,action='store_true',help="Stage 3: train classifier, frozen mask generator")
	parser.add_argument("--mask_add_aug",default=False,action='store_true',help="Augment the mask for stage 3")
	parser.add_argument("--stage4",default=False,action='store_true',help="Stage 4: Evaluate the model on standard GSC test set")
	parser.add_argument("--stage5",default=False,action='store_true',help="Stage 5: Evaluate the model on the GSC-musan test set")
	parser.add_argument("--stage6",default=False,action='store_true',help="Stage 6: Evaluate the model on the GSC-QUT test set")
	parser.add_argument("--stagedev",default=False,action='store_true',help="Evaluate the model on the dev set")
	parser.add_argument("--gen_augment",default=False,action='store_true',help="Adding noise to generator input")
	parser.add_argument("--musan_path",type=str,default="./data/musan_1s/noise/train/",help="Musan noise with 1 second length")
	parser.add_argument("--musan_sc_path",type=str,default="./data/SpeechCommands_Musan",help="path to in domain test set GSC-Musan ")
	parser.add_argument("--qut_sc_path",type=str,default="./data/SpeechCommands_QUT",help="path to out domain test set GSC-OUT directory")
	parser.add_argument("--constant_value",type=float,default=0.0001)
	parser.add_argument("--gen_aug_type",type=int,default=1)
	parser.add_argument("--mask_shift",type=int,default=10)
	parser.add_argument("--load_best_classifier_stage1",default=False,action='store_true',help="if True then load the best trained classifier for stage1")
	parser.add_argument("--pre_train_clf",type=str,default=None)
	parser.add_argument("--vanillaNoiseAug",default=False,action='store_true',help="vanilla noise augment")
	parser.add_argument("--target_SNR_dB",type=float,default=6)
	parser.add_argument("--q",type=float,default=5,help='binarized important Aug')
	parser.add_argument("--best_model_path",type=str,default=None)
	parser.add_argument("--musan_folder_name_base",type=str,default='speech_command_test_musan')
	parser.add_argument("--qut_folder_name_base",type=str,default='speech_command_test_QUT')
	parser.add_argument("--google_sc_test_txt_path",type=str,default='./data/SpeechCommands/speech_commands_v0.02/testing_list.txt')
	args = parser.parse_args()
	print('type(vars(args))',type(vars(args)))

	print('args',args)
	main(args)
