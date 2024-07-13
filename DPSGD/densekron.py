"""Created in May, 2020
Pytorch functions for damped preconditioned SGD
@author: Muhammad Usman Qadeer
"""
import torch

_tiny = 1e-38   # to avoid division by zero

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

from utils.Dense_kron_preconditioner import *

class dense_kron:
	def __init__(self, Ws, use_4D = False, use_1D = False, use_damping = False, lambd = 1, eta = 1e-3):
		self.use_4D = use_4D
		self.use_1D = use_1D
		self.params = Ws
		self.use_damping = use_damping
		self.lambd = lambd
		self.eta = eta
		
		
	def initialize_preconditioners(self):
		Qs = []
		for W in self.params:
			if len(W.shape) == 2:
				Qs.append([torch.eye(W.shape[0]).to(device), torch.eye(W.shape[1]).to(device)])
			elif len(W.shape) == 1:
				Qs.append([torch.eye(W.shape[0]).to(device)])
			elif len(W.shape) == 4:
				if self.use_4D:
					Qs.append([torch.eye(W.shape[0]).to(device), torch.eye(W.shape[1]).to(device), torch.eye(W.shape[2]).to(device), torch.eye(W.shape[3]).to(device)])
				else:
					Qs.append([torch.eye(W.shape[0]).to(device), torch.eye(W.shape[1]*W.shape[2]*W.shape[3]).to(device)])
		return Qs
		
	def initialize_preconditioner(self, W):
		
		if len(W.shape) == 2:
			q = [torch.eye(W.shape[0]).to(device), torch.eye(W.shape[1]).to(device)]
		elif len(W.shape) == 1:
			q = torch.eye(W.shape[0]).to(device)
		elif len(W.shape) == 4:
			if self.use_4D:
				q = [torch.eye(W.shape[0]).to(device), torch.eye(W.shape[1]).to(device), torch.eye(W.shape[2]).to(device), torch.eye(W.shape[3]).to(device)]
			else:
				q = [torch.eye(W.shape[0]).to(device), torch.eye(W.shape[1]*W.shape[2]*W.shape[3]).to(device)]
		return q
	
	def precondition_grads(self, q, g, lambd):
		s = g.shape
		if self.use_damping:
			if len(s) == 2:
				pG = precond_grad_kron_damped_2D(q[0], q[1], g, lambd, self.eta)
			elif len(s) == 1:
				if not self.use_1D:
					pG = g
				else:
					pG = precond_grad_kron_damped_1D(q, g, lambd, self.eta)
			elif len(s) == 4:
				if not self.use_4D:
					pG = precond_grad_kron_damped_2D(q[0], q[1], g.reshape(s[0],-1), lambd, self.eta).reshape(s)
				else:
					pG = precond_grad_kron_damped_4D(q[0], q[1], q[2], q[3], g, lambd, self.eta)
		else:
			if len(s) == 2:
				pG = precond_grad_kron_2D(q[0], q[1], g)
			elif len(s) == 1:
				if not self.use_1D:
					pG = g
				else:
					pG = precond_grad_kron_1D(q, g)
			elif len(s) == 4:
				if not self.use_4D:
					pG = precond_grad_kron_2D(q[0], q[1], g.reshape(s[0],-1)).reshape(s)
				else:
					pG = precond_grad_kron_4D(q[0], q[1], q[2], q[3], g)
		return pG
		
	def update_preconditioner(self, q, dX, dG, step = 0.01):
		s = dG.shape
		if len(s) == 2:
			new_q = update_precond_kron_2D(q[0], q[1], dX, dG, step=step)
		elif len(s) == 1:
			if not self.use_1D:
				new_q = q
			else:
				new_q = update_precond_kron_1D(q, dX, dG, step=step)
		elif len(s) == 4:
			if not self.use_4D:
				new_q = update_precond_kron_2D(q[0], q[1], dX.reshape(s[0],-1), dG.reshape(s[0],-1), step=step)
			else:
				new_q = update_precond_kron_4D(q[0], q[1], q[2], q[3], dX, dG, step=step)

		return new_q

