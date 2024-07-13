"""Created in May, 2020
Pytorch functions for damped preconditioned SGD
@author: Muhammad Usman Qadeer
"""
import torch

_tiny = 1e-38   # to avoid division by zero

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



############################################################    SCAW TYPE   ####################################################################
######################################## Update Scaw Type ###############################
def update_precond_scaw_1D(Ql, dG, dX, step=0.01):
    max_l = torch.max(torch.abs(Ql))
    rho = torch.sqrt(max_l)
    Ql = Ql/rho
    A = Ql.matmul(dG)
    Bt = (torch.triangular_solve(dX.reshape(-1,1), Ql.t(), upper=False))[0]
    grad1 = torch.triu(A.matmul(A.t()) - Bt.matmul(Bt.t()))
    step1 = step/(torch.max(torch.abs(grad1)) + _tiny)
    return Ql - step1*(grad1.matmul(Ql))
    
def update_precond_scaw_2D(Ql, qr, dX, dG, step=0.01):
    """
    update scaling-and-whitening preconditioner
    """
    max_l = torch.max(torch.abs(Ql))
    max_r = torch.max(torch.abs(qr))
    
    rho = torch.sqrt(max_l/max_r)
    Ql = Ql/rho
    qr = rho*qr
    
    A = Ql.mm( dG*qr )
    Bt = torch.triangular_solve(dX/qr, Ql.t(), upper=False)[0]
    
    grad1 = torch.triu(A.mm(A.t()) - Bt.mm(Bt.t()))
    grad2 = torch.sum(A*A, dim=0, keepdim=True) - torch.sum(Bt*Bt, dim=0, keepdim=True)
    
    step1 = step/(torch.max(torch.abs(grad1)) + _tiny)
    step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
        
    return Ql - step1*grad1.mm(Ql), qr - step2*grad2*qr
    
######################################## Preconditioned Gradient SCAW ###############################
def precond_grad_scaw_1D(Ql, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    Grad: (matrix) gradient
    """
    return Ql.t() @ Ql @ Grad
    
def precond_grad_scaw_2D(Ql, qr, Grad):
    """
    apply scaling-and-whitening preconditioner
    """

    return Ql.t().mm(Ql.mm(Grad*(qr*qr)))

################################# Preconditioned Gradient SCAW Damped ###############################
def precond_grad_scaw_damped_1D(Ql, Grad, lambd, eta):
    P1 = Ql.t().mm(Ql)
    pi = torch.trace(P1)
    preG = P1 @ Grad + (pi)*(eta + lambd**0.5)*Grad

    return preG

def precond_grad_scaw_damped_2D(Ql, qr, Grad, lambd, eta):
    P1 = Ql.t().mm(Ql)
    P2 = qr*qr
    pi = (torch.trace(P1)*P2.shape[0])/(torch.sum(P2)*P1.shape[0])
    IL = torch.ones(P1.shape[0]).to(device)
    P1 = P1 + torch.diag(torch.sqrt((pi)*(eta + lambd**0.5))*IL)
    P2 = P2 + torch.sqrt((1/pi)*(eta + lambd**0.5))
    return P1.mm(Grad*(P2))





# NOTE 1:
# SCAN preconditioner is super sparse, sparser than a diagonal preconditioner! 
# For an (M, N) matrix, it only requires 2*M+N-1 parameters to represent it
# Make sure that input feature vector is augmented by 1 at the end, and the affine transformation is given by
#               y = x*(affine transformation matrix)

# NOTE 1:
# We have defined 4D functions for convolutional neural networks in case of full dense kron type preconditioner
# others are sparser and 2D would work for them
