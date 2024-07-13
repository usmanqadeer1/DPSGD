function [cost0] = preconditioned_SGD_sparse()
batch_size = 100;
W1 = randn(100, 2+1)/sqrt(2+1); % initializing neural network coefficients 
W2 = randn(1, size(W1,1)+1)/sqrt(size(W1,1)+1);
max_iter = 1e4;
cost0 = zeros(1, max_iter);
Q1_left = eye(size(W1, 1)); % kron(Q1_right'*Q1_right, Q1_left'*Q1_left) will be the preconditioner for W1
Q1_right = eye(size(W1, 2));    
Q2 = eye(length(W2));   % Q2'*Q2 will be the preconditioner for W2
for iter = 1 : max_iter
    grad_W1 = zeros(size(W1));  % these ones will be stochastic gradients
    grad_W2 = zeros(size(W2));
    
    delta_W1 = sqrt(eps)*randn(size(W1));   % small perturbation of W
    delta_W2 = sqrt(eps)*randn(size(W2));   
    W1_ = W1 + delta_W1;    % perturbed W
    W2_ = W2 + delta_W2;
    grad_W1_ = zeros(size(W1)); % these ones will be the perturbed gradients
    grad_W2_ = zeros(size(W2));
    for i = 1 : batch_size
        u = rand(2, 1); % raw features
        v = sqrt(12)*(u - [0.5;0.5]);   % normalized features
        class_label = mod(round(10^u(1) - 10^u(2)), 2); % class label
        class_label = 2*class_label - 1;
        
        x1 = tanh( W1*[v; 1] ); % output of first layer
        y = W2*[x1;1];  % output of second layer
        e = log(1+exp(-class_label*y)); % logistic loss
        cost0(iter) = cost0(iter) + e;
        J = -class_label/(1+exp(class_label*y));
        grad_W2 = grad_W2 + J*[x1; 1]';
        J = (1-x1.*x1).*(W2(:,1:end-1)'*J);
        grad_W1 = grad_W1 + J*[v; 1]';
        
        x1 = tanh( W1_*[v; 1] );    % calculating the perturbed gradients
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
    
    [Q1_left, Q1_right] = preconditioner_kron(delta_W1, grad_W1_ - grad_W1, Q1_left, Q1_right); % adapt preconditioners 
    Q2 = preconditioner(delta_W2', grad_W2_' - grad_W2', Q2);
    
    W1 = W1 - 0.1*Q1_left'*Q1_left*grad_W1*Q1_right'*Q1_right;  % preconditioned SGD
    W2 = W2 - 0.1*(Q2'*Q2*grad_W2')';
    
    
    if mod(iter, 100) == 0
        fprintf('Iteration %g; loss %g \n', iter, mean(cost0(iter-99:iter)));
    end
end
%plot(cost0)

end