clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
                          
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

load('ex4weights.mat');

nn_params = [Theta1(:) ; Theta2(:)];

fprintf('\nFeedforward Using Neural Network ...\n')

lambda = 1;

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
J = 0;

Theta1_grad = zeros(size(Theta1));

Theta2_grad = zeros(size(Theta2));

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

J=sum(sum((-Y).*log(h)-(1-Y).*log(1-h),2))/m;

Theta_aux1=Theta1(:,2:end);
Theta_aux2=Theta2(:,2:end);
R_aux=sum(sum(Theta_aux1.^2))+sum(sum(Theta_aux2.^2));
%+sum(sum(Theta_aux2.^2));
R=(lambda*R_aux)/(2*m);
J_reg=J+R;
            