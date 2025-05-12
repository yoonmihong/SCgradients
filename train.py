import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
# from preprocessing import *
from model import *
import torch.optim as optim
# from collections import OrderedDict
import csv
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from pytorchtools import EarlyStopping
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import resreg
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from torch.utils.tensorboard import SummaryWriter


criterion = nn.L1Loss()
criterion_smooth = nn.SmoothL1Loss(beta=0.2)
criterion_mse = nn.MSELoss(reduction='sum')
criterion_class = F.nll_loss#nn.CrossEntropyLoss()#F.nll_loss#nn.BCELoss()#
pdist = nn.PairwiseDistance(p=2)
criterion_GAN = nn.BCELoss()#nn.MSELoss()


#estimate inter-pFC for regularization
def compute_corr_loss(gen_sample,batch_size):
	intra_corr = 0
	count = 0
	for i in range(batch_size):
		start = gen_sample[i, :]
		# start = start[torch.triu(torch.ones(start.shape), diagonal=1) == 1] # vectorize upper triangular part of matrix
		for j in range(batch_size):
			if (j!=i):
				count +=1
				temp=gen_sample[j,:]
				# temp = temp[torch.triu(torch.ones(temp.shape), diagonal=1) == 1] # vectorize upper triangular part of matrix
				# corr = correlation_coefficient_loss(start, temp)
				corr = pearson_correlation(start, temp)#1 - correlation_loss(start, temp)
				intra_corr += corr
	return intra_corr/count

def compute_corr_siamese(gen_sample,target,batch_size):
	intra_corr = 0
	count = 0
	for i in range(batch_size):
		start = gen_sample[i, :]
		start = start[torch.triu(torch.ones(start.shape), diagonal=1) == 1] # vectorize upper triangular part of matrix
		start_target = target[i, :]
		start_target = start_target[torch.triu(torch.ones(start_target.shape), diagonal=1) == 1]
		for j in range(batch_size):
			if (j!=i):
				count +=1
				temp=gen_sample[j,:]
				temp = temp[torch.triu(torch.ones(temp.shape), diagonal=1) == 1] # vectorize upper triangular part of matrix
				# corr = correlation_coefficient_loss(start, temp)
				corr = pearson_correlation(start, temp)#1 - correlation_loss(start, temp)
				temp_target=target[j,:]
				temp_target = temp_target[torch.triu(torch.ones(temp_target.shape), diagonal=1) == 1]
				corr_target = pearson_correlation(start_target, temp_target)
				error = torch.abs(corr-corr_target)*(corr-corr_target)
				intra_corr += error
	return intra_corr/count

def compute_mse_siamese(gen_sample,target):
	intra_corr = 0
	count = 0
	batch_size = target.shape[0]
	for i in range(batch_size):
		start = gen_sample[i, :]
		start_target = target[i, :]
		for j in range(batch_size):
			if j>i:#(j!=i):
				count +=1
				
				temp=gen_sample[j,:]
				error_gen = F.mse_loss(temp, start)
				temp_target=target[j,:]
				error_target = F.mse_loss(temp_target, start_target)
				error = F.mse_loss(error_gen, error_target)
				intra_corr += error
	return intra_corr/count


def pearson_correlation(output, target):
	vx = output - torch.mean(output) + 1e-6
	vy = target - torch.mean(target) + 1e-6

	# cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)+1e-8) * torch.sqrt(torch.sum(vy ** 2)+1e-8))
	cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) )
	return cost

def correlation_loss(output, target):
	res = pearson_correlation(output, target)
	return (1-res)*(1-res)



def train(model, X_train, training_adj, y_train,  X_test, test_adj, y_test, minval, maxval, num_CV, args):
	
			
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay, amsgrad=True)

	learning_scheduler = ExponentialLR(optimizer, gamma=args.gamma)

	print_network(model)
	all_epochs_loss = []
	# initialize the early_stopping object
	save_filename = 'train_checkpoint_CV'+ str(num_CV)+ '.pt' 
	save_path = os.path.join(args.save_dir, args.log_name, save_filename)
	if not os.path.exists(os.path.join(args.save_dir, args.log_name)):
		os.makedirs(os.path.join(args.save_dir, args.log_name))



	mean_score = np.mean(y_train)
	print ("Target mean score: ", mean_score)
	mean_score = torch.tensor([mean_score]).cuda()
	summary_filename = 'CV'+ str(num_CV)
	writer = SummaryWriter(log_dir=os.path.join(args.save_dir, args.log_name, summary_filename))

	BatchSetSrc = np.zeros([args.batch_size, args.lr_dim*(args.grad_comp_SC)], dtype=np.float32)
	BatchSetAdj = np.zeros([args.batch_size, args.lr_dim, args.lr_dim], dtype=np.float32)
	BatchSetTar = np.zeros([args.batch_size, 1], dtype=np.float32)

	# for siamese network
	BatchSetSrc1 = np.zeros([args.batch_size, args.lr_dim*(args.grad_comp_SC)], dtype=np.float32)
	BatchSetAdj1 = np.zeros([args.batch_size, args.lr_dim, args.lr_dim], dtype=np.float32)
	BatchSetTar1 = np.zeros([args.batch_size, 1], dtype=np.float32)

	BatchSetSrc2 = np.zeros([args.batch_size, args.lr_dim*(args.grad_comp_SC)], dtype=np.float32)
	BatchSetAdj2 = np.zeros([args.batch_size, args.lr_dim, args.lr_dim], dtype=np.float32)
	BatchSetSrc3 = np.zeros([args.batch_size, args.lr_dim*(args.grad_comp_SC)], dtype=np.float32)
	BatchSetAdj3 = np.zeros([args.batch_size, args.lr_dim, args.lr_dim], dtype=np.float32)
	for epoch in range(args.epochs):
		with torch.autograd.set_detect_anomaly(True):

			epoch_loss = []
			epoch_recon_loss = []
			epoch_siamese_loss = []
			epoch_reg_loss = []
			bCount = 0

			# np.random.seed(seed=epoch)
			shuffler = np.random.permutation(len(X_train))
			X_train_shuffled = np.array(X_train)[shuffler]
			Adj_train_shuffled = np.array(training_adj)[shuffler]
			y_train_shuffled = np.array(y_train)[shuffler]

			# for siamese network
			# np.random.seed(seed=epoch+100)
			shuffler1 = np.random.permutation(len(X_train))
			X_train_shuffled1 = np.array(X_train)[shuffler1]
			Adj_train_shuffled1 = np.array(training_adj)[shuffler1]
			y_train_shuffled1 = np.array(y_train)[shuffler1]

			shuffler2 = np.random.permutation(len(X_train))
			X_train_shuffled2 = np.array(X_train)[shuffler2]
			Adj_train_shuffled2 = np.array(training_adj)[shuffler2]
			shuffler3 = np.random.permutation(len(X_train))
			X_train_shuffled3 = np.array(X_train)[shuffler3]
			Adj_train_shuffled3 = np.array(training_adj)[shuffler3]
			for p in range(len(X_train)):

				Input = X_train_shuffled[p]
				Adj = Adj_train_shuffled[p]		
				Output = y_train_shuffled[p]	

				# for siamese network
				Input1 = X_train_shuffled1[p]	
				Adj1 = Adj_train_shuffled1[p]	
				Output1 = y_train_shuffled1[p]

				Input2 = X_train_shuffled2[p]
				Adj2 = Adj_train_shuffled2[p]	
				Input3 = X_train_shuffled3[p]
				Adj3 = Adj_train_shuffled3[p]		
				Input = np.array(Input)
				Adj = np.array(Adj)
				Output = np.array(Output)

				# for siamese network
				Input1 = np.array(Input1)
				Adj1 = np.array(Adj1)
				Output1 = np.array(Output1)


				Input2 = np.array(Input2)
				Adj2 = np.array(Adj2)
				Input3 = np.array(Input3)
				Adj3 = np.array(Adj3)
				if args.noise_std > 0:
					noise_std = args.noise_std # 0.1

					noise = np.random.normal(0, noise_std, 2)

					Output = Output + noise[0]
					Output1 = Output1 + noise[1]

				BatchSetSrc[bCount, :] = Input
				BatchSetAdj[bCount, :] = Adj
				BatchSetTar[bCount, :] = Output

				# for siamese network
				BatchSetSrc1[bCount, :] = Input1
				BatchSetAdj1[bCount, :] = Adj1
				BatchSetTar1[bCount, :] = Output1


				BatchSetSrc2[bCount, :] = Input2
				BatchSetAdj2[bCount, :] = Adj2
				BatchSetSrc3[bCount, :] = Input3
				BatchSetAdj3[bCount, :] = Adj3
				bCount +=1
				if bCount < args.batch_size-1:
					continue
				# print (BatchSetScanner)
				Input = torch.from_numpy(BatchSetSrc).type(torch.FloatTensor)
				Adj = torch.from_numpy(BatchSetAdj).type(torch.FloatTensor)
				Output = torch.from_numpy(BatchSetTar).type(torch.FloatTensor)

				# for siamese network
				Input1 = torch.from_numpy(BatchSetSrc1).type(torch.FloatTensor)
				Adj1 = torch.from_numpy(BatchSetAdj1).type(torch.FloatTensor)
				Output1 = torch.from_numpy(BatchSetTar1).type(torch.FloatTensor)

				Input2 = torch.from_numpy(BatchSetSrc2).type(torch.FloatTensor)
				Adj2 = torch.from_numpy(BatchSetAdj2).type(torch.FloatTensor)
				Input3 = torch.from_numpy(BatchSetSrc3).type(torch.FloatTensor)
				Adj3 = torch.from_numpy(BatchSetAdj3).type(torch.FloatTensor)
				if torch.cuda.is_available():
					Input = Input.cuda()
					Adj = Adj.cuda()
					Output = Output.cuda()

					# for siamese network
					Input1 = Input1.cuda()
					Adj1 = Adj1.cuda()
					Output1 = Output1.cuda()

					Input2 = Input2.cuda()
					Adj2 = Adj2.cuda()
					Input3 = Input3.cuda()
					Adj3 = Adj3.cuda()
				# for positive 
				if args.corruption > 0:
					corruption_mask = torch.zeros_like(Input, dtype=torch.bool).cuda()
					corruption_mask1 = torch.zeros_like(Input1, dtype=torch.bool).cuda()
					corruption_len = int(args.corruption*args.lr_dim*(args.grad_comp_SC))
					for i in range(args.batch_size):
						corruption_idx = torch.randperm(args.lr_dim*(args.grad_comp_SC))[:corruption_len]
						corruption_mask[i, corruption_idx] = True
						corruption_idx1 = torch.randperm(args.lr_dim*(args.grad_comp_SC))[:corruption_len]
						corruption_mask1[i, corruption_idx1] = True
						# corruption_mask[i,:] = corruption_mask[i,:] + corruption_mask[i,:].t()
					Input2 = torch.where(corruption_mask, Input2, Input).cuda()
					Input3 = torch.where(corruption_mask1, Input3, Input1).cuda()
				

				model.train()
				optimizer.zero_grad()

				net_out = model(Input, Adj)
				# for siamese network
				net_out1 = model(Input1, Adj1)
				if args.corruption > 0:
					net_out2 = model(Input2, Adj)
					net_out3 = model(Input3, Adj1)


				recon_loss = F.mse_loss(net_out, Output)  
				# # for siamese network
				recon_loss += F.mse_loss(net_out1, Output1)
				if args.corruption > 0:
					recon_loss += F.mse_loss(net_out2, Output) + F.mse_loss(net_out3, Output1)

				# for siamese loss
				siamese_loss = F.mse_loss(Output1.unsqueeze(2)-Output.unsqueeze(1), net_out1.unsqueeze(2)-net_out.unsqueeze(1))
				if args.corruption > 0:
					siamese_loss += F.mse_loss(Output1.unsqueeze(2)-Output.unsqueeze(1), net_out3.unsqueeze(2)-net_out2.unsqueeze(1))
				generator_loss = recon_loss  + args.lam_siamese*siamese_loss
				generator_loss.backward()
				optimizer.step()
				# optimizer.step(closure)

				epoch_loss.append(generator_loss.item())
				epoch_recon_loss.append(recon_loss.item())
				epoch_siamese_loss.append(siamese_loss.item())

				bCount = 0
				BatchSetSrc.fill(0)
				BatchSetAdj.fill(0)
				BatchSetTar.fill(0)

				# for siamese network
				BatchSetSrc1.fill(0)
				BatchSetAdj1.fill(0)
				BatchSetTar1.fill(0)
				BatchSetSrc2.fill(0)
				BatchSetAdj2.fill(0)
				BatchSetSrc3.fill(0)
				BatchSetAdj3.fill(0)
			# validation==test
			output_val = []
			netout_val = []
			valid_losses = []
			scanner_accuracy = 0
			# for p in range(pLength, len(subjects_adj)):
			for p in range(len(X_test)):
				model.eval()
				bCount = 0
				BatchSetSrc.fill(0)
				BatchSetAdj.fill(0)
				BatchSetTar.fill(0)
				
				with torch.no_grad():
					Input = X_test[p]	
					Adj = test_adj[p]
					Output = y_test[p]	

					Input = np.array(Input)
					Adj = np.array(Adj)
					Output = np.array(Output)

					BatchSetSrc[bCount, :] = Input
					BatchSetAdj[bCount, :] = Adj
					BatchSetTar[bCount, :] = Output
					

					Input = torch.from_numpy(BatchSetSrc).type(torch.FloatTensor)
					Adj = torch.from_numpy(BatchSetAdj).type(torch.FloatTensor)
					Output = torch.from_numpy(BatchSetTar).type(torch.FloatTensor)
				

					if torch.cuda.is_available():
						Input = Input.cuda()
						Adj = Adj.cuda()
						Output = Output.cuda()


					net_out = model(Input, Adj)
					net_out = net_out[0]
					Output = Output[0]
					error = criterion_mse(net_out, Output).mean()

					valid_losses.append(error.item())					
					output_val.append(Output)#.detach().cpu().float().numpy().astype(np.float32)*(maxval-minval) + minval)
					netout_val.append(net_out)#.detach().cpu().float().numpy().astype(np.float32)*(maxval-minval) + minval)

			valid_loss = np.average(valid_losses)
			val_std = np.std(valid_losses)
			
			# learning_scheduler.step(valid_loss) # for ReduceLROnPlateau
			learning_scheduler.step() 
			if epoch%20==0:
				lr = optimizer.param_groups[0]['lr']
				corr = pearson_correlation(torch.FloatTensor(netout_val), torch.FloatTensor(output_val))

				print("Epoch: ", epoch, 'LR: %.7f' % lr, "recon_loss: %.4f" %np.mean(epoch_recon_loss),"siamese_loss: %.4f" %np.mean(epoch_siamese_loss))
				print("Epoch: ", epoch, "Val mse: %.4f (%.3f)" %(np.mean(valid_losses), np.std(torch.FloatTensor(netout_val).cpu().float().numpy().astype(np.float32))  ), "Val corr: %.3f"%corr)
			all_epochs_loss.append(np.mean(epoch_loss))

			writer.add_scalar('Loss/train', np.mean(epoch_loss), epoch)
			writer.add_scalar('Loss/recon', np.mean(epoch_recon_loss), epoch)
			writer.add_scalar('Loss/siamese', np.mean(epoch_siamese_loss), epoch)
			writer.add_scalar('Loss/validation_mse', np.mean(valid_losses), epoch)
			writer.add_scalars('Error',{'train_error':np.mean(epoch_loss),'val_error':np.mean(valid_losses)},  epoch)

			
			writer.add_scalar('Learning rate/generator', optimizer.param_groups[0]['lr'], epoch)

			torch.save(model.state_dict(), save_path)
			
			writer.close()
	# test:
	# # load the last checkpoint with the best model
	model.load_state_dict(torch.load(save_path))
	# print ("Test at epoch: ", epoch)
	filedir = os.path.join(args.log_path, args.log_name)
	if not os.path.exists(os.path.join(args.log_path, args.log_name)):
		os.makedirs(os.path.join(args.log_path, args.log_name))
	


	test_losses = []
	scanner_accuracy = 0
	pred_arr = []
	gt_arr = []
	# for p in range(pLength, len(subjects_adj)):
	for p in range(len(X_test)):
		model.eval()
		bCount = 0
		BatchSetSrc.fill(0)
		BatchSetAdj.fill(0)
		BatchSetTar.fill(0)

		with torch.no_grad():
			Input = X_test[p]	
			Adj = test_adj[p]
			Output = y_test[p]	

			Input = np.array(Input)
			Adj = np.array(Adj)
			Output = np.array(Output)

			BatchSetSrc[bCount, :] = Input
			BatchSetAdj[bCount, :] = Adj
			BatchSetTar[bCount, :] = Output

			Input = torch.from_numpy(BatchSetSrc).type(torch.FloatTensor)
			Adj = torch.from_numpy(BatchSetAdj).type(torch.FloatTensor)
			Output = torch.from_numpy(BatchSetTar).type(torch.FloatTensor)
			
			if torch.cuda.is_available():
				Input = Input.cuda()
				Adj = Adj.cuda()
				Output = Output.cuda()
				
			
			net_out = model(Input, Adj)
			net_out = net_out[0]
			Output = Output[0]

			error = criterion_mse(net_out, Output).mean()


			test_losses.append(error.item())

			
			# renormalize:
			
			pred_res = net_out*(maxval-minval) + minval
			GT_res = y_test[p]*(maxval-minval) + minval
			pred_res = pred_res.squeeze().detach().cpu().float().numpy().astype(np.float32)
			GT_res = GT_res.astype(np.float32)


			filename = filedir +'/CV'+ str(num_CV)+'.txt'
			with open(filename, "a") as log_file:
				log_file.write('%s,%s\n' % (pred_res, GT_res))
			print (pred_res, GT_res)

			pred_arr.append(pred_res)
			gt_arr.append(GT_res)
	# print (pred_arr)
	# print (gt_arr)
	corr, pval = pearsonr(np.stack(pred_arr), np.stack(gt_arr))
	print ("CV"+str(num_CV) +"_Test recon error: %.4f (+-%.3f)" %(np.mean(test_losses), np.std(test_losses)))
	print ("CV"+str(num_CV) +"_Test correlation:  %.4f, %.4f " %(corr, pval))
	return np.mean(test_losses), pred_arr, gt_arr, corr
