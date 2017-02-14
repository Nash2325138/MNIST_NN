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

% Trasform Y from 1~10 to binary vector
Y = zeros(size(y,1), num_labels);
for i = 1:size(Y,1)
	Y(i, y(i)) = 1;
end


% Forward prop
A1 = [ones(size(X,1), 1), X];

Z2 = A1 * Theta1';
A2 = [ones(size(A1,1), 1), sigmoid(Z2)];

Z3 = A2 * Theta2';
A3 = sigmoid(Z3);


% Calculate cost
H = A3;
J = sum(sum(-Y .* log(H) - (1-Y) .* log(1 - H), 2)) / m;

J += sum(sum(Theta1(:,2:end) .^ 2, 2)) * lambda / m / 2;
J += sum(sum(Theta2(:,2:end) .^ 2, 2)) * lambda / m / 2;


% Backward prop
A3_grad = A3 - Y; % Ai_grad doesn't have bias term, but Ai does
A2_grad = (A3_grad * Theta2(:,2:end)) .* sigmoidGradient(Z2);

delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));
for i = 1:m
	delta_1 += A2_grad(i, :)' * A1(i, :);
	delta_2 += A3_grad(i, :)' * A2(i, :);
end
Theta1_grad = delta_1 ./ m;
Theta2_grad = delta_2 ./ m;

% Backward prop with regulation
Theta1_grad(:, 2:end) += lambda .* Theta1(:, 2:end) ./ m;
Theta2_grad(:, 2:end) += lambda .* Theta2(:, 2:end) ./ m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
