function Q = preconditioner(dx, dg, Q)
% Solving preconditioner by minimizing cost 
%   dg'*P*dg + dx'*inv(P)*dx
% with chol decomposition P = Q'*Q
rho = sqrt(max(abs(dx)) * max(abs(dg)));
if rho == 0
    return;
end
dx = dx/rho;
dg = dg/rho; 
step_size = 0.01;
term1 = Q*dg;
term2 = dx'/Q;
grad = term1*term1' - term2'*term2; 
grad = triu( grad );
Q = Q - step_size*grad*Q/(max(max(abs(grad)))+eps);