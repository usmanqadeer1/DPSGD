"""Created in May, 2020
Pytorch functions for damped preconditioned SGD
@author: Muhammad Usman Qadeer
"""
import torch

_tiny = 1e-38   # to avoid division by zero

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


######################################## Update Kron Dense ###############################
def update_precond_kron_1D(Ql, dX, dG, step=0.01):

    max_l = torch.max(torch.abs(Ql))
    rho = torch.sqrt(max_l)
    Ql = Ql/rho

    A = Ql.matmul(dG)
    Bt = (torch.triangular_solve(dX.reshape(-1,1), Ql.t(), upper=False))[0]
    grad1 = torch.triu(A.matmul(A.t()) - Bt.matmul(Bt.t()))

    step1 = step/(torch.max(torch.abs(grad1)) + _tiny)
    return Ql - step1*(grad1.matmul(Ql))

def update_precond_kron_2D(Ql, Qr, dX, dG, step=0.01):
    """
    update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Ql: (left side) Cholesky factor of preconditioner with positive diagonal entries
    Qr: (right side) Cholesky factor of preconditioner with positive diagonal entries
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: normalized step size in range [0, 1] 
    """
    max_l = torch.max(torch.abs(Ql))
    max_r = torch.max(torch.abs(Qr))
    
    rho = torch.sqrt(max_l/max_r)
    Ql = Ql/rho
    Qr = rho*Qr
    
    A = Ql.mm( dG.mm( Qr.t() ) )
    Bt = torch.triangular_solve((torch.triangular_solve(dX.t(), Qr.t(), upper=False))[0].t(), Ql.t(), upper=False)[0]
    
    grad1 = torch.triu(A.mm(A.t()) - Bt.mm(Bt.t()))
    grad2 = torch.triu(A.t().mm(A) - Bt.t().mm(Bt))
    
    step1 = step/(torch.max(torch.abs(grad1)) + _tiny)
    step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
        
    return Ql - step1*grad1.mm(Ql), Qr - step2*grad2.mm(Qr)

# def update_precond_kron_4D(Ql, Qr, Qb, Qd, dX, dG, step=0.01):
#     """
#     update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
#     Ql: (left side) Cholesky factor of preconditioner with positive diagonal entries
#     Qr: (right side) Cholesky factor of preconditioner with positive diagonal entries
#     Qb: (back side) Cholesky factor of preconditioner with positive diagonal entries
#     Qd: (depth side) Cholesky factor of preconditioner with positive diagonal entries
#     dX: perturbation of (matrix) parameter
#     dG: perturbation of (matrix) gradient
#     step: normalized step size in range [0, 1] 
#     """
#     max_l = torch.max(torch.abs(Ql))
#     max_r = torch.max(torch.abs(Qr))
#     max_b = torch.max(torch.abs(Qb))
#     max_d = torch.max(torch.abs(Qd))
    
#     rho = torch.sqrt(max_l/max_r)
#     dho = torch.sqrt(max_b/max_d)
    
#     Ql = Ql/rho
#     Qr = rho*Qr
#     Qb = Qb/dho
#     Qd = dho*Qd
    
#     At =  (Qr @ (Qb @ dG @ Qd.T).T @ Ql.T).T
#     Bt = (torch.inverse(Qr) @ (torch.inverse(Qb) @ dX @ torch.inverse(Qd.T)).T @ torch.inverse(Ql.T)).T

#     a, b, c, d = At.shape
#     A, B = At.reshape(a, -1), Bt.reshape(a, -1)
#     grad1 = torch.triu(A.mm(A.t()) - B.mm(B.t()))
#     A, B = At.reshape(b, -1), Bt.reshape(b, -1)
#     grad2 = torch.triu(A.mm(A.t()) - B.mm(B.t()))
#     A, B = At.reshape(c, -1), Bt.reshape(c, -1)
#     grad3 = torch.triu(A.mm(A.t()) - B.mm(B.t()))
#     A, B = At.reshape(d, -1), Bt.reshape(d, -1)
#     grad4 = torch.triu(A.mm(A.t()) - B.mm(B.t()))

#     step1 = step/(torch.max(torch.abs(grad1)) + _tiny)
#     step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
#     step3 = step/(torch.max(torch.abs(grad3)) + _tiny)
#     step4 = step/(torch.max(torch.abs(grad4)) + _tiny)
        
#     return Ql - step1*grad1.mm(Ql), Qr - step2*grad2.mm(Qr), Qb - step3*grad3.mm(Qb), Qd - step4*grad4.mm(Qd)

def update_precond_kron_4D(Ql, Qr, Qb, Qd, dX, dG, step=0.01):
    """
    update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Ql: (left side) Cholesky factor of preconditioner with positive diagonal entries
    Qr: (right side) Cholesky factor of preconditioner with positive diagonal entries
    Qb: (back side) Cholesky factor of preconditioner with positive diagonal entries
    Qd: (depth side) Cholesky factor of preconditioner with positive diagonal entries
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: normalized step size in range [0, 1] 
    """
    max_l = torch.max(torch.abs(Ql))
    max_r = torch.max(torch.abs(Qr))
    max_b = torch.max(torch.abs(Qb))
    max_d = torch.max(torch.abs(Qd))
    max_r = max(max_r, max_b, max_d)
    
    rho = torch.sqrt(max_l/max_r)
    
    Ql = Ql/rho
    Qr = rho*Qr
    Qb = rho*Qb
    Qd = rho*Qd
    
    At =  (Qr @ (Qb @ dG @ Qd.T).T @ Ql.T).T
    Bt = (torch.inverse(Qr) @ (torch.inverse(Qb) @ dX @ torch.inverse(Qd.T)).T @ torch.inverse(Ql.T)).T

    a, b, c, d = At.shape
    A, B = At.reshape(a, -1), Bt.reshape(a, -1)
    grad1 = torch.triu(A.mm(A.t()) - B.mm(B.t()))
    A, B = At.reshape(b, -1), Bt.reshape(b, -1)
    grad2 = torch.triu(A.mm(A.t()) - B.mm(B.t()))
    A, B = At.reshape(c, -1), Bt.reshape(c, -1)
    grad3 = torch.triu(A.mm(A.t()) - B.mm(B.t()))
    A, B = At.reshape(d, -1), Bt.reshape(d, -1)
    grad4 = torch.triu(A.mm(A.t()) - B.mm(B.t()))

    step1 = step/(torch.max(torch.abs(grad1)) + _tiny)
    step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
    step3 = step/(torch.max(torch.abs(grad3)) + _tiny)
    step4 = step/(torch.max(torch.abs(grad4)) + _tiny)
        
    return Ql - step1*grad1.mm(Ql), Qr - step2*grad2.mm(Qr), Qb - step3*grad3.mm(Qb), Qd - step4*grad4.mm(Qd)

######################################## Preconditioned Gradient Kron ###############################
def precond_grad_kron_1D(Ql, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    Grad: (matrix) gradient
    """
    return Ql.t() @ Ql @ Grad
    
def precond_grad_kron_2D(Ql, Qr, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    Grad: (matrix) gradient
    """
    if Grad.shape[0] > Grad.shape[1]:
        return Ql.t().mm( Ql.mm( Grad.mm( Qr.t().mm(Qr) ) ) )
    else:
        return (((Ql.t().mm(Ql)).mm(Grad)).mm(Qr.t())).mm(Qr)
        
def precond_grad_kron_4D(Ql, Qr, Qb, Qd, Grad):
    P1 = Ql.t().mm(Ql)
    P2 = Qr.t().mm(Qr)
    P3 = Qb.t().mm(Qb)
    P4 = Qd.t().mm(Qd)
    
    return (P2 @ (P3 @ Grad @ P4.T).T @ P1.T).T

######################################## Preconditioned Gradient Kron Damped ###############################
def precond_grad_kron_damped_1D(Ql, Grad,lambd, eta):
    P1 = Ql.t().mm(Ql)
    pi = torch.trace(P1)
    IL = torch.ones(P1.shape[0]).to(device)
    P1 = P1 + torch.diag(torch.sqrt((pi)*(eta + lambd**0.5))*IL)

    return P1.matmul(Grad)

def precond_grad_kron_damped_2D(Ql, Qr, Grad,lambd, eta):
    P1 = Ql.t().mm(Ql)
    P2 = Qr.t().mm(Qr)
    pi = (torch.trace(P1)*P2.shape[0])/(torch.trace(P2)*P1.shape[0])
    IL = torch.ones(P1.shape[0]).to(device)
    IR = torch.ones(P2.shape[0]).to(device)
    P1 = P1 + torch.diag(torch.sqrt((pi)*(eta + lambd**0.5))*IL)
    P2 = P2 + torch.diag(torch.sqrt((1/pi)*(eta + lambd**0.5))*IR)
#     P1 = P1 + torch.diag(pi*((eta + lambd)**0.5)*IL)
#     P2 = P2 + torch.diag((1/pi)*((eta + lambd)**0.5)*IR)

    return P1.mm(Grad).mm(P2)

# def precond_grad_kron_damped_4D(Ql, Qr, Qb, Qd, Grad,lambd, eta):
#     P1 = Ql.t().mm(Ql)
#     P2 = Qr.t().mm(Qr)
#     P3 = Qb.t().mm(Qb)
#     P4 = Qd.t().mm(Qd)

#     pi = (torch.trace(P1)*P2.shape[0])/(torch.trace(P2)*P1.shape[0])
#     pi2 = (torch.trace(P3)*P4.shape[0])/(torch.trace(P4)*P3.shape[0])

#     IL = torch.ones(P1.shape[0]).to(device)
#     IR = torch.ones(P2.shape[0]).to(device)
#     IB = torch.ones(P3.shape[0]).to(device)
#     ID = torch.ones(P4.shape[0]).to(device)
#     P1 = P1 + torch.diag(torch.sqrt((pi)*(eta + lambd**0.5)*IL))
#     P2 = P2 + torch.diag(torch.sqrt((1/pi)*(eta + lambd**0.5)*IR))
#     P3 = P3 + torch.diag(torch.sqrt((pi2)*(eta + lambd**0.5)*IB))
#     P4 = P4 + torch.diag(torch.sqrt((1/pi2)*(eta + lambd**0.5)*ID))
    
#     return (P2 @ (P3 @ Grad @ P4.T).T @ P1.T).T

def precond_grad_kron_damped_4D(Ql, Qr, Qb, Qd, Grad,lambd, eta):
    P1 = Ql.t().mm(Ql)
    P2 = Qr.t().mm(Qr)
    P3 = Qb.t().mm(Qb)
    P4 = Qd.t().mm(Qd)
    
    shape2 = (P2.shape[0] + P3.shape[0] + P4.shape[0])
    trace2 = torch.trace(P2) + torch.trace(P3) + torch.trace(P4) 
    pi = (torch.trace(P1)*shape2)/(trace2*P1.shape[0])

    IL = torch.ones(P1.shape[0]).to(device)
    IR = torch.ones(P2.shape[0]).to(device)
    IB = torch.ones(P3.shape[0]).to(device)
    ID = torch.ones(P4.shape[0]).to(device)
    
    P1 = P1 + torch.diag(torch.sqrt((pi)*(eta + lambd**0.25))*IL)
    P2 = P2 + torch.diag(torch.sqrt((1/pi)*(eta + lambd**0.25))*IR)
    P3 = P3 + torch.diag(torch.sqrt((pi)*(eta + lambd**0.25))*IB)
    P4 = P4 + torch.diag(torch.sqrt((1/pi)*(eta + lambd**0.25))*ID)
    
    return (P2 @ (P3 @ Grad @ P4.T).T @ P1.T).T
    
	

	

