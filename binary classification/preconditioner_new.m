function [cost0] = preconditioner_new()
% this demo shows preconditioned SGD using a dense preconditioner
batch_size = 100;
W1 = randn(100, 2+1)/sqrt(2+1); % neural network coefficients 
W2 = randn(1, size(W1,1)+1)/sqrt(size(W1,1)+1);
max_iter = 1e5;
cost0 = zeros(1, max_iter);
P_dim = size(W1,1)*size(W1,2) + size(W2,1)*size(W2,2);  % dimension of preconditioner
gamma = 0.3;
Q = eye(P_dim);
Qold = zeros(P_dim); 
tic
for iter = 1 : max_iter
    grad_W1 = zeros(size(W1));
    grad_W2 = zeros(size(W2));
    
    delta_W1 = sqrt(eps)*randn(size(W1));   % perturbation of coefficients, and perturbed gradients
    delta_W2 = sqrt(eps)*randn(size(W2));
    W1_ = W1 + delta_W1;
    W2_ = W2 + delta_W2;
    grad_W1_ = zeros(size(W1));
    grad_W2_ = zeros(size(W2));
    for i = 1 : batch_size
        u = rand(2, 1); % raw features
        v = sqrt(12)*(u - [0.5;0.5]);   % normalized features
        class_label = mod(round(10^u(1) - 10^u(2)), 2);
        class_label = 2*class_label - 1;
        
        x1 = tanh( W1*[v; 1] );
        y = W2*[x1;1];
        e = log( 1 + exp(-class_label*y) ); % logisitic loss
        cost0(iter) = cost0(iter) + e;
        J = -class_label/(1+exp(class_label*y));
        grad_W2 = grad_W2 + J*[x1; 1]';
        J = (1-x1.*x1).*(W2(:,1:end-1)'*J);
        grad_W1 = grad_W1 + J*[v; 1]';
        
        x1 = tanh( W1_*[v; 1] );    % this part calculates the perturbed gradients
        y = W2_*[x1;1];
        J = -class_label/(1+exp(class_label*y));
        grad_W2_ = grad_W2_ + J*[x1; 1]';
        J = (1-x1.*x1).*(W2_(:,1:end-1)'*J);
        grad_W1_ = grad_W1_ + J*[v; 1]';
    end
    cost0(iter) = cost0(iter) / batch_size;
    
    grad_W1 = grad_W1/batch_size;
    grad_W2 = grad_W2/batch_size;
    
    grad_W1_ = grad_W1_/batch_size;
    grad_W2_ = grad_W2_/batch_size;
    
    Q = preconditioner([delta_W1(:);delta_W2(:)], [grad_W1_(:)-grad_W1(:); grad_W2_(:)-grad_W2(:)], Q); % solving for a preconditioner for all these coefficients
    Qnew = ((1-gamma) * Q + gamma * Qold)/ (1- gamma);
    delta_x = -0.1*Qnew'*Qnew*[grad_W1(:); grad_W2(:)];   % preconditioned SGD    
    n = 0;
    W1 = W1 + reshape(delta_x(n+1:n+size(W1,1)*size(W1,2)), size(W1,1), size(W1,2));
    n = n+size(W1,1)*size(W1,2);
    W2 = W2 + reshape(delta_x(n+1:n+size(W2,1)*size(W2,2)), size(W2,1), size(W2,2));
    Qold = Q;
    if mod(iter, 100) == 0
        fprintf('Iteration %g; loss %g \n', iter, mean(cost0(iter-99:iter)));
    end
end
%plot(cost0)

end