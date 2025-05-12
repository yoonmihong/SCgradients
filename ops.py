# from curses import A_COLOR
# from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# from layers import *
import scipy.sparse as sp

from torch_geometric.nn import DenseGraphConv, InstanceNorm, GraphNorm
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import global_mean_pool
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
# from scipy.linalg import eigvals
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.stats import truncnorm
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from enum import Enum
import numpy as np

class ChebGraphConv(nn.Module):
	def __init__(self, K, in_features, out_features, bias=True):
		super(ChebGraphConv, self).__init__()
		self.K = K
		self.weight = nn.Parameter(torch.FloatTensor(K+1, in_features, out_features))
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.in_features = in_features
		self.out_features = out_features
		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		# torch.nn.init.xavier_uniform(self.weight)
		if self.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
			torch.nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, x, gso):
		# Chebyshev polynomials:
		# x_0 = x,
		# x_1 = gso * x,
		# x_k = 2 * gso * x_{k-1} - x_{k-2},
		# where gso = 2 * gso / eigv_max - id.

		cheb_poly_feat = []
		if self.K < 0:
			raise ValueError('ERROR: The order of Chebyshev polynomials shoule be non-negative!')
		elif self.K == 0:
			# x_0 = x
			cheb_poly_feat.append(x)
		elif self.K == 1:
			# x_0 = x
			cheb_poly_feat.append(x)
			if gso.is_sparse:
				# x_1 = gso * x
				cheb_poly_feat.append(torch.sparse.mm(gso, x))
			else:
				if x.is_sparse:
					x = x.to_dense
				# x_1 = gso * x
				# cheb_poly_feat.append(torch.mm(gso, x))
				cheb_poly_feat.append(torch.matmul(gso, x))
		else:
			# x_0 = x
			cheb_poly_feat.append(x)
			if gso.is_sparse:
				# x_1 = gso * x
				cheb_poly_feat.append(torch.sparse.mm(gso, x))
				# x_k = 2 * gso * x_{k-1} - x_{k-2}
				for k in range(2, self.K+1):
					cheb_poly_feat.append(torch.sparse.mm(2 * gso, cheb_poly_feat[k - 1]) - cheb_poly_feat[k - 2])
			else:
				if x.is_sparse:
					x = x.to_dense
				# x_1 = gso * x
				cheb_poly_feat.append(torch.matmul(gso, x)) ## may have different results (not deterministic)
				# x_k = 2 * gso * x_{k-1} - x_{k-2}
				for k in range(2, self.K+1):
					cheb_poly_feat.append(torch.matmul(2 * gso, cheb_poly_feat[k - 1]) - cheb_poly_feat[k - 2])
		# print (cheb_poly_feat.shape)
		# feature = torch.stack(cheb_poly_feat, dim=0)
		feature = torch.stack(cheb_poly_feat, dim=1)
		# print (feature.shape)
		if feature.is_sparse:
			feature = feature.to_dense()
		# cheb_graph_conv = torch.einsum('bij,bjk->ik', feature, self.weight)
		cheb_graph_conv = torch.einsum('bdij,djk->bik', feature, self.weight)

		if self.bias is not None:
			# cheb_graph_conv = torch.add(input=cheb_graph_conv, other=self.bias, alpha=1)
			cheb_graph_conv = cheb_graph_conv + self.bias
		else:
			cheb_graph_conv = cheb_graph_conv

		return cheb_graph_conv

	def extra_repr(self) -> str:
		return 'K={}, in_features={}, out_features={}, bias={}'.format(
			self.K, self.in_features, self.out_features, self.bias is not None
		)
