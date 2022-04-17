import torch
import torch.optim as optim

def optimizer_helper(model,args):
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	if args.cosine_annealing:
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,50, eta_min=0, last_epoch=-1, verbose=False)
	else:
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) 
	return optimizer,scheduler