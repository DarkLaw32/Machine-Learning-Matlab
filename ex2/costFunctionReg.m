function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z=X*theta;
h=sigmoid(z);
one=ones(m,1);
aux1=-(y.*(log(h)))-((one-y).*log(one-h));
aux2=sum(aux1);
theta_aux=[0;theta(2:size(theta),:)];
aux3=theta_aux.^2;R=lambda*sum(aux3)/(2*m);
J=(aux2/m)+R;




[a,b]=size(grad);
aux3=(h-y).*X;
grad_ini=(1/m)*sum(aux3);

for j=1:a
    if j==1
        grad(j)=grad_ini(j);    
    else
        grad(j)=grad_ini(j)+(lambda/m)*theta(j); 
    end
end




% =============================================================

end
