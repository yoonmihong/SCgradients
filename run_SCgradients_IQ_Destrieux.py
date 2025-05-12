
import torch
# torch.use_deterministic_algorithms(True)
torch.set_deterministic(True)
import os
import random
import numpy as np
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
os.environ["PYTHONHASHSEED"] = str(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)

from model import *
from train import *
import argparse
from sklearn.model_selection import StratifiedKFold, KFold
import networkx as nx
import itertools
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import eigvals
from neuroCombat import neuroCombat
import pandas as pd
from neurocombat_sklearn import CombatModel
from brainspace.gradient import GradientMaps
from numpy import linalg
from scipy.sparse import csgraph
from brainspace.gradient.embedding import diffusion_mapping, laplacian_eigenmaps
from sklearn.metrics import pairwise_distances
import random
from brainspace.gradient.kernels import compute_affinity
from scipy.spatial.distance import pdist, squareform
import time
import json

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=300, metavar='no_epochs',
				help='number of episode to train ')
	parser.add_argument('--lr', type=float, default=0.005, metavar='lr',
				help='learning rate (default: 0.0001 using Adam Optimizer)')
	parser.add_argument('--beta1', type=float, default=0.9, 
				help='coefficients used for computing running averages of gradient (default: 0.9 using Adam Optimizer)')
	parser.add_argument('--weight_decay', type=float, default=0.0005,
					help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--noise_std', type=float, default=0.1,
					help='Gaussian noise for GT score (default: 0.1)')
	parser.add_argument('--corruption', type=float, default=0.1,
					help='Corruption rate for contrastive learning (default: 0.5)')
	parser.add_argument('--inputnoise_std', type=float, default=0.1,
					help='Gaussian noise for input (default: 0.1)')
	parser.add_argument('--step_size', type=int, default=1,
					help='step size for learning_scheduler')
	parser.add_argument('--gamma', type=float, default=0.99,
					help='decay for learning_scheduler')
	parser.add_argument('--lam_siamese', type=float, default=10.0, 
				help='Regularization hyperparameter')
	parser.add_argument('--lam_recon', type=float, default=10.0, 
				help='Regularization hyperparameter')
	parser.add_argument('--K', type=int, default=3, metavar='N',
				help='Chebyshev polynomial order')
	parser.add_argument('--lr_dim', type=int, default=148, metavar='N',
				help='adjacency matrix input dimensions')
	parser.add_argument('--hr_dim', type=int, default=90, metavar='N',
				help='super-resolved adjacency matrix output dimensions')
	parser.add_argument('--batch_size', type=int, default=10, metavar='N',
				help='batch sizes')
	parser.add_argument('--log_name', type=str, default='TestError', 
				help='file name to save errors')
	parser.add_argument('--optimizer', type=str, default='ADAM', 
				help='optimizer')
	parser.add_argument('--log_path', type=str, default='./Results_IQ/', 
				help='file name to save errors')
	parser.add_argument('--save_dir', type=str, default='./Checkpoints/', 
				help='file name to save errors')
	parser.add_argument('--SCyear', type=int, default=1, metavar='N',
				help='age for SC')
	parser.add_argument('--score', type=str, default='FSIQ4', 
				help='target cognitive score')
	parser.add_argument('--random_state', type=int, default=1, metavar='N',
				help='random state for data splitting')
	parser.add_argument('--grad_comp_SC', type=int, default=2, metavar='N',
				help='number of gradient components for input')
	parser.add_argument('--sparsity_SC', type=float, default=0, metavar='N',
				help='sparsity level for SC gradients')
	parser.add_argument('--grad_app', type=str, default='dm', metavar='N',
				help='Gradient maps approach in BrainSpace')
	parser.add_argument('--grad_kernel', type=str, default='normalized_angle', metavar='N',
				help='Gradient maps kernel in BrainSpace')
	args = parser.parse_args()

	

	filedir = os.path.join(args.log_path, args.log_name)
	if not os.path.exists(os.path.join(args.log_path, args.log_name)):
		os.makedirs(os.path.join(args.log_path, args.log_name))
	## save argparse
	file_arg = filedir + '/commandline_args.txt'
	with open(file_arg, 'w') as f:
		json.dump(args.__dict__, f, indent=2)

	def normalize(mx):
		"""Row-normalize sparse matrix"""
		rowsum = np.array(mx.sum(1))
		r_inv = np.power(rowsum, -1).flatten()
		r_inv[np.isinf(r_inv)] = 0.
		r_mat_inv = sp.diags(r_inv)
		mx = r_mat_inv.dot(mx)
		return mx

	def normalize_sym(adj):
		row_sum = np.sum(adj, axis=1)
		row_sum_inv_sqrt = np.power(row_sum, -0.5)
		row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
		deg_inv_sqrt = np.diag(row_sum_inv_sqrt)
		# A_{sym} = D^{-0.5} * A * D^{-0.5}
		sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)
		return sym_norm_adj

	def calc_chebynet_gso(gso):
		# if sp.issparse(gso):
		#     id = sp.identity(gso.shape[0], format='csc')
		#     eigval_max = max(eigsh(A=gso, k=6, which='LM', return_eigenvectors=False))
		# else:
		id = np.identity(gso.shape[0])
		eigval_max = max(eigvals(a=gso).real)
		
		# If the gso is symmetric or random walk normalized Laplacian,
		# then the maximum eigenvalue has to be smaller than or equal to 2.
		if eigval_max >= 2:
			gso = gso - id
		else:
			gso = 2 * gso / eigval_max - id

		return gso	
	
	def generate_symmetric_matrix(size, min_val=0, max_val=1):
		"""
		Generate a random symmetric matrix.
		
		Args:
		size (int): The size of the square matrix (number of rows/columns)
		min_val (float, optional): Minimum value for random elements. Defaults to 0.
		max_val (float, optional): Maximum value for random elements. Defaults to 1.
		
		Returns:
		numpy.ndarray: A symmetric matrix
		"""
		# Generate a random matrix
		matrix = np.random.uniform(min_val, max_val, (size, size))
		
		# Make the matrix symmetric by adding the matrix to its transpose and dividing by 2
		symmetric_matrix = (matrix + matrix.T) / 2
		
		# fill diagonal zero
		np.fill_diagonal(symmetric_matrix, 0)
		
		return symmetric_matrix
	
	# compute SC gradients from SC matrices
	SC_list = [generate_symmetric_matrix(size=148) for _ in range(100)]
	if args.grad_comp_SC > 0:

		grad_components = 20
		gref = GradientMaps(approach=args.grad_app, kernel=args.grad_kernel, random_state=0, n_components=grad_components)

		n = 0
		gref.fit(SC_list[n], sparsity=args.sparsity_SC)

		gp = GradientMaps(approach=args.grad_app, kernel=args.grad_kernel, alignment='procrustes', random_state=0, n_components=grad_components)

		gp.fit(SC_list, sparsity=args.sparsity_SC, n_iter=500, reference=gref.gradients_)

		
		gradients_procrustes_SC = np.zeros([len(SC_list), args.lr_dim, args.grad_comp_SC], dtype=np.float32)
		lambdas = [None] * len(SC_list)
		for i in range(len(SC_list)):
			for c in range((args.grad_comp_SC)):

				gradients_procrustes_SC[i,:,c] = np.abs(gp.aligned_[i][:,c])

			lambdas[i] = gp.lambdas_[i]

		gradients_procrustes_SC = gradients_procrustes_SC[:,:,:args.grad_comp_SC]

	# Generate random variables from normal distribution
	IQ_list = np.random.normal(loc=100, scale=15, size=100)
	# compute mean and std
	meanval = np.mean(IQ_list)
	stdval = np.std(IQ_list)

		
	# # # normalize IQ
	minval = np.min(np.array(IQ_list))
	maxval = np.max(np.array(IQ_list))
	print ("Min score: ", minval, " Max score: ", maxval)
	

	minval = meanval - 3.0*stdval
	maxval = meanval + 3.0*stdval
	IQ_list = (np.array(IQ_list) - minval)/(maxval-minval) 
	IQ_list[IQ_list<0.0] = 0.0
	IQ_list[IQ_list>1.0] = 1.0
	



	bins = np.linspace(0, len(IQ_list), 10)
	y_binned = np.digitize(IQ_list, bins)

	GT = []
	pred = []


	r = args.random_state
	print ("random state: ", r)
	skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=r)

	idx = np.linspace(0,len(SC_list)-1,len(SC_list),dtype=int)

	score_group = np.zeros(len(SC_list), dtype=np.int32)
	print ("Total number of samples: ", len(SC_list))
	
	num_CV = 0
	pred_list = []
	gt_list = []

	grad_list = []
	for i in range(len(SC_list)):
		content = (SC_list[i])

		# # row-wise normalization
		content = normalize(content)
		# # Kipf: renomalization trick
		content = content + np.identity(content.shape[0])
		content = normalize_sym(content)
		# # Laplacian
		content = np.identity(content.shape[0]) - content
		# # Laplacian rescale
		content = calc_chebynet_gso(content)
		SC_list[i] = content 
		grad_list.append(np.hstack((gradients_procrustes_SC[i].flatten())))
	starttime = time.time()
	for train_index, test_index in skf.split(SC_list, y_binned):

		X_train = [grad_list[index] for index in train_index]
		A_train = [SC_list[index] for index in train_index]
		y_train = [IQ_list[index] for index in train_index]

		X_test = [grad_list[index] for index in test_index]
		A_test = [SC_list[index] for index in test_index]
		y_test = [IQ_list[index] for index in test_index]


		model = SCgradients_IQ_GCN_sppool(args)
		if torch.cuda.is_available():
			device = 'cuda'
			model.to(device)

			score, pred, gt, test_corr = train(model, X_train, A_train, y_train,  X_test, A_test, y_test,  minval, maxval, num_CV, args)
			pred_list.append(pred)
			gt_list.append(gt)

		num_CV +=1

	endtime = time.time()
	print ("All CV finished..%.3f" %(endtime-starttime))
	pred_list = np.concatenate(pred_list).ravel()
	gt_list = np.concatenate(gt_list).ravel()
	pred_list = np.stack(np.asarray(pred_list, dtype=np.float32)).ravel()
	gt_list = np.stack(np.asarray(gt_list, dtype=np.float32)).ravel()

	filename = filedir + '/pred_summary.csv'
	with open(filename, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['Prediction', 'GT', 'subj_id'])
		# writer.writerows(someiterable)
		for s in range(len(pred_list)):
			writer.writerow([pred_list[s], gt_list[s]])

	corr, pval = pearsonr(gt_list, pred_list)
	print ("All Test recon error: %.4f " %(np.mean(np.abs(gt_list-pred_list))))
	print ("All Test correlation:  %.4f, %.4f " %(corr, pval))
	corr, pval = spearmanr(gt_list, pred_list)
	print ("All Test Spearman:  %.4f, %.4f " %(corr, pval))

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print ("Interrupted")