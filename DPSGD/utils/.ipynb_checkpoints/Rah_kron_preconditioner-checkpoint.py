"""Created in May, 2020
Pytorch functions for damped preconditioned SGD
@author: Muhammad Usman Qadeer
"""
import torch
_tiny = 1e-38   # to avoid division by zero

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
################################### update functions ###############################################
def update_precond_rah_1D(ql, dX, dG, step=0.01):
	max_l = torch.max(torch.abs(ql))
	rho = torch.sqrt(max_l)
	ql = ql / rho
	
	A = ql[0:1].t() * dG
	A[:-1,:] = A[:-1,:] + ql[1:,:-1].t() * dG[-1:]
	
	Bt = (1.0/ql[0:1].t())*dX
	Bt[-1:] = Bt[-1:] -(ql[1:,:-1]/(ql[0:1,:-1]*ql[0,-1])) @ dX[:-1] # Ql^(-T)*dX
    
	# update Ql
	grad1_diag = torch.sum(A * A, axis=1) - torch.sum(Bt * Bt, axis=1)
	grad1_bias = A[:-1].mm(A[-1:].t()) - Bt[:-1].mm(Bt[-1:].t())
	grad1_bias = torch.reshape(grad1_bias, [-1])
	grad1_bias = torch.cat([grad1_bias, grad1_bias.new_zeros(1)])

	
	step1 = step / (torch.maximum(torch.max(torch.abs(grad1_diag)), torch.max(torch.abs(grad1_bias))) + _tiny)
	new_ql0 = ql[0] - step1 * grad1_diag * ql[0]
	new_ql1 = ql[1] - step1 * (grad1_diag * ql[1] + ql[0, -1] * grad1_bias)

	return torch.stack((new_ql0, new_ql1))

def update_precond_rah_2D(ql, qr, dX, dG, step=0.01):
    
    max_l = torch.max(torch.abs(ql))
    max_r = torch.max(torch.abs(qr))
    rho = torch.sqrt(max_l / max_r)
    ql = ql / rho
    qr = rho * qr

    a = ql[0:1].t() * dG
    a[:-1,:] = a[:-1,:] + ql[1:,:-1].t() * dG[-1:]
    A = a * qr[0:1]
    A[:,:-1] = A[:,:-1] +a[:,-1:].mm(qr[1:,:-1]) #Ql*dG*Qr^T

    bt = (1.0/ql[0:1].t())*dX
    bt[-1:] = bt[-1:] -(ql[1:,:-1]/(ql[0:1,:-1]*ql[0,-1])) @ dX[:-1]
    Bt = bt*(1/qr[0:1])
    Bt[:,-1:] = Bt[:,-1:] - bt[:,:-1] @ (qr[1:,:-1]/(qr[0:1,:-1]*qr[0,-1])).T# Ql^(-T)*dX*Qr^(-1)

    # update Ql
    grad1_diag = torch.sum(A * A, axis=1) - torch.sum(Bt * Bt, axis=1)
    grad1_bias = A[:-1].mm(A[-1:].t()) - Bt[:-1].mm(Bt[-1:].t())
    grad1_bias = torch.reshape(grad1_bias, [-1])
    grad1_bias = torch.cat([grad1_bias, grad1_bias.new_zeros(1)])

    step1 = step / (torch.maximum(torch.max(torch.abs(grad1_diag)), torch.max(torch.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1 * grad1_diag * ql[0]
    new_ql1 = ql[1] - step1 * (grad1_diag * ql[1] + ql[0, -1] * grad1_bias)

    # update Qr
    grad2_diag = torch.sum(A * A, axis=0) - torch.sum(Bt * Bt, axis=0)
    grad2_bias = A[:,:-1].t() @ A[:,-1] - Bt[:,:-1].t() @ (Bt[:,-1])
    grad2_bias = torch.reshape(grad2_bias, [-1])
    grad2_bias = torch.cat([grad2_bias, grad2_bias.new_zeros(1)])

    step2 = step / (torch.maximum(torch.max(torch.abs(grad2_diag)), torch.max(torch.abs(grad2_bias))) + _tiny)
    new_qr0 = qr[0] - step2 * grad2_diag * qr[0]
    new_qr1 = qr[1] - step2 * (grad2_diag * qr[1] + qr[0, -1] * grad2_bias)
    return torch.stack((new_ql0, new_ql1)), torch.stack((new_qr0, new_qr1))


##########s############################################## Normal (SCAN Pro) type  ########################################################################
def precond_grad_rah_1D(ql, Grad):

	preG = ql[0:1].t() * Grad
	preG[:-1,:] = preG[:-1,:] + ql[1:,:-1].t() * Grad[-1:] # Ql*Grad
	
	add_last_row = ql[1:,:-1].mm(preG[:-1])
	preG = ql[0:1].t() * preG
	preG[-1:] = preG[-1:] + add_last_row #Ql^T*Ql*dG
	
	return preG

def precond_grad_rah_2D(ql, qr, Grad):

	preG = ql[0:1].t() * Grad
	preG[:-1,:] = preG[:-1,:] + ql[1:,:-1].t() * Grad[-1:] # Ql*Grad
	
	add_last_row = ql[1:,:-1].mm(preG[:-1])
	preG = ql[0:1].t() * preG
	preG[-1:] = preG[-1:] + add_last_row #Ql^T*Ql*dG
	
	A = preG
	preG = preG * qr[0:1]
	preG[:,:-1] = preG[:,:-1] + A[:,-1:].mm(qr[1:,:-1]) #Ql^T*Ql*dG*Qr^T
	
	A = preG
	preG = preG * qr[0:1]
	preG[:,-1:] = preG[:,-1:] + A[:,:-1].mm(qr[1:,:-1].t())
	return preG

##########s############################################## Normal Damped(SCAN Pro) type  ########################################################################
def precond_grad_rah_damped_1D(ql, Grad, lambd, eta):
	preG = ql[0:1].t() * Grad
	preG[:-1,:] = preG[:-1,:] + ql[1:,:-1].t() * Grad[-1:] # Ql*Grad
	
	add_last_row = ql[1:,:-1].mm(preG[:-1])
	preG = ql[0:1].t() * preG
	preG[-1:] = preG[-1:] + add_last_row #Ql^T*Ql*dG
	
	return preG

def precond_grad_rah_damped_2D(ql, qr, Grad, lambd, eta):
    trace1 = torch.sum(ql[0:1]**2) + torch.sum(ql[1:,:-1]**2)
    trace2 = torch.sum(qr[0:1]**2) + torch.sum(qr[1:,:-1]**2)
    pi = (trace1*qr.shape[1])/(trace2*ql.shape[1])
    
    P1_diag = ql[0:1]**2
    P1_diag[:,-1] = P1_diag[:,-1] + torch.sum(ql[1:,:-1]**2)
    P1_bias = ql[0:1,:-1]*ql[1:,:-1]
    P1_diag = P1_diag + torch.sqrt((pi)*(eta + lambd**0.5))
    
    P2_diag = qr[0:1]**2
    P2_diag[:,-1] = P2_diag[:,-1] + torch.sum(qr[1:,:-1]**2)
    P2_bias = qr[0:1,:-1]*qr[1:,:-1]
    P2_diag = P2_diag + torch.sqrt((1/pi)*(eta + lambd**0.5))
    
    preG1 = P1_diag.t() * Grad
    preG1[:-1,:] = preG1[:-1,:] + P1_bias.t().mm(Grad[-1:])
    preG1[-1:] = preG1[-1:] + P1_bias.mm(Grad[:-1]) #Ql^T*Ql*dG
    preG = preG1 * P2_diag
    preG[:,:-1] = preG[:,:-1] + preG1[:,-1:].mm(P2_bias)
    preG[:,-1:] = preG[:,-1:] + preG1[:,:-1].mm(P2_bias.t())
    return preG





# NOTE 1:
# SCAN preconditioner is super sparse, sparser than a diagonal preconditioner! 
# For an (M, N) matrix, it only requires 2*M+N-1 parameters to represent it
# Make sure that input feature vector is augmented by 1 at the end, and the affine transformation is given by
#               y = x*(affine transformation matrix)

# NOTE 1:
# We have defined 4D functions for convolutional neural networks in case of full dense kron type preconditioner
# others are sparser and 2D would work for them
