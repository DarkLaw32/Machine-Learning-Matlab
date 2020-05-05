function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Parte 1 del examen
I=eye(num_labels);%%Crea una matriz identidad de 10x10
Y=zeros(m, num_labels);%%Crea una matriz de ceros de 5000x10 una columna por cada salida en la ultima capa

for i=1:m%%Crea una con unos con las soluciones de la salida
    Y(i,:)=I(y(i),:);
end

%Calculo de h(x)
a1=[ones(m,1) X];
z2=a1*Theta1';
a2=[ones(size(z2, 1), 1) sigmoid(z2)];%Anadimos una columna de unos debido a las bias de activacion
z3=a2*Theta2';
a3=sigmoid(z3);
h=a3;

J_aux=sum(sum((-Y).*log(h)-(1-Y).*log(1-h),2))/m;

%SEGUNDA PARTE COSTO CON REGULARIZACION
Theta_aux1=Theta1(:,2:end);%Creamos un vector auxiliar de Theta1 eliminando la primera columna correspondiente a bias.
Theta_aux2=Theta2(:,2:end);%Creamos un vector auxiliar de Theta2 eliminando la primera columna correspondiente a bias.
R_aux=sum(sum(Theta_aux1.^2,2))+sum(sum(Theta_aux2.^2,2));%Calculamos el valor de R sin las constantes
R=(lambda*R_aux)/(2*m);%Inclusion de las constantes de la formula
J=J_aux+R;%Calculo del costo con y sin regularizacion

%TERCERA PARTE CALCULO DEL GRADIENTE
sigma3=a3-Y;%Segundo paso: Calculo de sigma 3 correspondiente a la salida
sigma2=(sigma3*Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]);%Tercer Paso Calculamos el valor de sigma2 anadiendo una fila de unos por las bias de activacion
sigma2=sigma2(:, 2:end);
delta1=sigma2'*a1;%Calculo de delta para la etapa 1
delta2=sigma3'*a2;%Calculo de delta para la etapa 2
R1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
R2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
%Valores de Theta con y sin regularizacion
Theta1_grad=(1/m)*delta1+R1;
Theta2_grad=(1/m)*delta2+R2;

%Calculo del Gradiente Total
grad = [Theta1_grad(:) ; Theta2_grad(:)];












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
