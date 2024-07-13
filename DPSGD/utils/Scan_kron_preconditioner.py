"""Created in May, 2020
Pytorch functions for damped preconditioned SGD
@author: Muhammad Usman Qadeer
"""
import torch

_tiny = 1e-38   # to avoid division by zero

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def update_precond_scan_1D(ql, dX, dG, step=0.01):
	max_l = torch.max(torch.abs(ql))
	rho = torch.sqrt(max_l)
	ql = ql / rho
	A = ql[0:1].t() * dG
	A[:-1,:] = A[:-1,:] + ql[1:,:-1].t() * dG[-1:]
    
	Bt = (1.0/ql[0:1].t())*dX
	Bt[-1:] = Bt[-1:] - (ql[1:,:-1]/(ql[0:1,:-1]*ql[0,-1])) @ dX[:-1]
    
	grad1_diag = torch.sum(A * A, axis=1) - torch.sum(Bt * Bt, axis=1)
	grad1_bias = A[:-1].mm(A[-1:].t()) - Bt[:-1].mm(Bt[-1:].t())
	grad1_bias = torch.reshape(grad1_bias, [-1])
	grad1_bias = torch.cat([grad1_bias, grad1_bias.new_zeros(1)])
    
	
	step1 = step / (torch.maximum(torch.max(torch.abs(grad1_diag)), torch.max(torch.abs(grad1_bias))) + _tiny)
	new_ql0 = ql[0] - step1 * grad1_diag * ql[0]
	new_ql1 = ql[1] - step1 * (grad1_diag * ql[1] + ql[0, -1] * grad1_bias)
    
	return torch.stack((new_ql0, new_ql1))


def update_precond_scan_2D(ql, qr, dX, dG, step=0.01):
	max_l = torch.max(torch.abs(ql))
	max_r = torch.max(torch.abs(qr))
	rho = torch.sqrt(max_l / max_r)
	ql = ql / rho
	qr = rho * qr
	
	A = ql[0:1].t() * dG
	A[:-1,:] = A[:-1,:] + ql[1:,:-1].t() * dG[-1:]
	A = A * qr
	
	Bt = (1.0/ql[0:1].t())*dX
	Bt[-1:] = Bt[-1:] -(ql[1:,:-1]/(ql[0:1,:-1]*ql[0,-1])) @ dX[:-1]
	Bt = Bt*(1.0/qr) # Ql^(-T)*dX*Qr^(-1) 
    
	# update Ql
	grad1_diag = torch.sum(A * A, axis=1) - torch.sum(Bt * Bt, axis=1)
	grad1_bias = A[:-1].mm(A[-1:].t()) - Bt[:-1].mm(Bt[-1:].t())
	grad1_bias = torch.reshape(grad1_bias, [-1])
	grad1_bias = torch.cat([grad1_bias, grad1_bias.new_zeros(1)])

	
	step1 = step / (torch.maximum(torch.max(torch.abs(grad1_diag)), torch.max(torch.abs(grad1_bias))) + _tiny)
	new_ql0 = ql[0] - step1 * grad1_diag * ql[0]
	new_ql1 = ql[1] - step1 * (grad1_diag * ql[1] + ql[0, -1] * grad1_bias)

	# update qr
	grad2 = torch.sum(A * A, dim=0, keepdim=True) - torch.sum(Bt * Bt, dim=0, keepdim=True)
	step2 = step / (torch.max(torch.abs(grad2)) + _tiny)
	new_qr = qr - step2 * grad2 * qr
	return torch.stack((new_ql0, new_ql1)), new_qr

############################ Preconditioned Gradient SCAN ############@###########################
	
def precond_grad_scan_1D(ql, Grad):
    preG = ql[0:1].t() * Grad
    preG[:-1,:] = preG[:-1,:] + ql[1:,:-1].t() * Grad[-1:] # Ql*Grad
    
    add_last_row = ql[1:,:-1].mm(preG[:-1])
    preG = ql[0:1].t() * preG
    preG[-1:] = preG[-1:] + add_last_row #Ql^T*Ql*dG
    return preG


def precond_grad_scan_2D(ql, qr, Grad):
    preG = ql[0:1].t() * Grad
    preG[:-1,:] = preG[:-1,:] + ql[1:,:-1].t() * Grad[-1:] # Ql*Grad
    
    add_last_row = ql[1:,:-1].mm(preG[:-1])
    preG = ql[0:1].t() * preG
    preG[-1:] = preG[-1:] + add_last_row #Ql^T*Ql*dG
    preG = preG * (qr*qr) #Ql^T*Ql*dG*qr*qr
    return preG

############################ Preconditioned Gradient SCAN Damped ############@###########################
def precond_grad_scan_damped_1D(ql, Grad, lambd, eta):
    preG = ql[0:1].t() * Grad
    preG[:-1,:] = preG[:-1,:] + ql[1:,:-1].t() * Grad[-1:] # Ql*Grad
    
    add_last_row = ql[1:,:-1].mm(preG[:-1])
    preG = ql[0:1].t() * preG
    preG[-1:] = preG[-1:] + add_last_row #Ql^T*Ql*dG
    return preG


def precond_grad_scan_damped_2D(ql, qr, Grad, lambd, eta):
    
    P2 = qr*qr
    trace1 = torch.sum(ql[0:1]**2) + torch.sum(ql[1:,:-1]**2)
    pi = (trace1*qr.shape[1])/(torch.sum(P2)*ql.shape[1])
    P2 = P2 + torch.sqrt((1/pi)*(eta + lambd**0.5))
    preG = ql[0:1].t() * Grad
    preG[:-1,:] = preG[:-1,:] + ql[1:,:-1].t() * Grad[-1:] # Ql*Grad
    
    add_last_row = ql[1:,:-1].mm(preG[:-1])
    preG = ql[0:1].t() * preG
    preG[-1:] = preG[-1:] + add_last_row #Ql^T*Ql*dG
    
    preG = preG + torch.sqrt((pi)*(eta + lambd**0.5))*Grad #(Ql^T*Ql + damping*I)dG = Ql^T*Ql*dG + damping*dG
    preG = preG * P2 #(Ql^T*Ql + damping*I)*dG*(qr*qr + damping*I)
    return preG

# in case of damped functions, sparse multiplications work better

