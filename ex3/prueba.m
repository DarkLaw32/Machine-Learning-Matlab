clear ; close all; clc
load('ex3data1.mat'); % training data stored in arrays X, y

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
m = length(y_t);
lambda_t = 3;
z=X_t*theta_t;
h = 1.0 ./ (1.0 + exp(-z));

one=ones(m,1);
aux1=-(y_t.*(log10(h)))-((one-y_t).*log10(one-h))
J=sum(aux1)/m

aux2=sum(theta_t.^2);
R=(lambda_t/(2*m))*aux2;

