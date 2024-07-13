"""Created in May, 2020
Pytorch functions for Shampoo
@author: Muhammad Usman Qadeer
"""
import torch

_tiny = 1e-38   # to avoid division by zero

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
def matrix_power(matrix, power):
    # use CPU for svd for speed up
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).cuda()
    
    


class Shampoo:
	def __init__(self, Ws, use_1D = False, use_damping = False, epsilon = 1e-4):
		self.use_1D = use_1D
		self.params = Ws
		self.use_damping = use_damping
		self.epsilon = epsilon
		
        
	def initialize_preconditioners(self):
		Qs = []
		for W in self.params:
			if len(W.shape) == 2:
				Qs.append([self.epsilon*torch.eye(W.shape[0]).to(device), self.epsilon*torch.eye(W.shape[1]).to(device)])
			elif len(W.shape) == 1:
				Qs.append([self.epsilon*torch.eye(W.shape[0]).to(device)])
			elif len(W.shape) == 4:
				Qs.append([self.epsilon*torch.eye(W.shape[0]).to(device), self.epsilon*torch.eye(W.shape[1]*W.shape[2]*W.shape[3]).to(device)])
		return Qs
		
	def initialize_preconditioner(self, W):
		
		if len(W.shape) == 2:
			q = [self.epsilon*torch.eye(W.shape[0]).to(device), self.epsilon*torch.eye(W.shape[1]).to(device)]
		elif len(W.shape) == 1:
			q = self.epsilon*torch.eye(W.shape[0]).to(device)
		elif len(W.shape) == 4:
			q = [self.epsilon*torch.eye(W.shape[0]).to(device), self.epsilon*torch.eye(W.shape[1]*W.shape[2]*W.shape[3]).to(device)]
		return q
	
	def precondition_grads(self, q, g):
		s = g.shape
		
		if len(s) == 2:
			pG = matrix_power(q[0], -1/4) @ g @ matrix_power(q[1], -1/4)
		elif len(s) == 1:
			if not self.use_1D:
				pG = g
			else:
				pG = matrix_power(q, -1/2) @ g
		elif len(s) == 4:
			pG = (matrix_power(q[0], -1/4) @ g.reshape(s[0], -1) @ matrix_power(q[1], -1/4)).reshape(s)
		return pG
		
	def update_preconditioner(self, q, g):
		s = g.shape
		if len(s) == 2:
			new_q = [q[0] + (g @ g.T), q[1] + (g.T @ g)]
		elif len(s) == 1:
			if not self.use_1D:
				new_q = q
			else:
				new_q = q + (g @ g.T)
		elif len(s) == 4:
			new_g = g.reshape(s[0], -1)
			new_q = [q[0] + (new_g @ new_g.T), q[1] + (new_g.T @ new_g)]
		return new_q
