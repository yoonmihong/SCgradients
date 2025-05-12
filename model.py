import torch
import torch.nn as nn
from ops import *
import torch.nn.functional as F


class SCgradients_IQ_GCN_sppool(nn.Module):

	def __init__(self, args):
		super(SCgradients_IQ_GCN_sppool, self).__init__()
		self.lr_dim = args.lr_dim
		self.hr_dim = args.hr_dim
		self.batch_size = args.batch_size
		self.K = args.K
		
		in_dim = args.grad_comp_SC 
		dim1 = 	32#16
		dim2 = 32#4
		dim3 = 16
		# self.drop = nn.Dropout(p=0.2)
		self.gcn1 = ChebGraphConv(self.K, in_dim, dim1)

		self.gcn2 = ChebGraphConv(self.K, dim1, dim2)

		self.net = nn.Sequential(
							nn.Linear(int(dim1*2+dim2*2),16),						
							nn.LeakyReLU(0.5),							
							)
		self.net_reg = nn.Linear(16,1)

	def forward(self, in_grad, in_SC, is_training=True):
		with torch.autograd.set_detect_anomaly(True):
			if torch.cuda.is_available():
				in_grad = in_grad.cuda()
				in_SC = in_SC.cuda()

			in_grad = in_grad.view(self.batch_size, self.lr_dim,-1)
	

			X = self.gcn1(in_grad, in_SC)
			X = nn.LeakyReLU(0.5, True)(X)

			X0 = X.view(self.batch_size, self.lr_dim, -1)
			Z1 = torch.mean(X0, dim=1, keepdim=True) 
			Z2,_ = torch.max(X0, dim=1, keepdim=True)
			Z = torch.cat([ Z1, Z2],2)

			X = self.gcn2(X, in_SC)	
			X = nn.LeakyReLU(0.5, True)(X)

			X = X.contiguous().view(self.batch_size, self.lr_dim,-1) + X0
			X0 = X.view(self.batch_size, self.lr_dim, -1)
			Z1 = torch.mean(X0, dim=1, keepdim=True) 
			Z2,_ = torch.max(X0, dim=1, keepdim=True)
			Z = torch.cat([Z,Z1,Z2],2)


			Z = Z.view(self.batch_size,-1)

			output = self.net(Z)

			output_reg = self.net_reg(output)

		return output_reg

def print_network(net):
	num_params = 0
	# for param in net.parameters():
	for name, param in net.state_dict().items():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)