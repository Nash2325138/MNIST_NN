function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, ...
                                   X, Y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, Y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
end1 = hidden_layer1_size * (input_layer_size + 1);
end2 = end1 + hidden_layer2_size * (hidden_layer1_size + 1);

Theta1 = reshape(nn_params(1:end1), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + end1):end2), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(nn_params((1 + end2):end), ...
                 num_labels, (hidden_layer2_size + 1));

m = size(X, 1);

% Forward prop
A1 = [ones(m, 1), X];

Z2 = A1 * Theta1';
A2 = [ones(m, 1), sigmoid(Z2)];

Z3 = A2 * Theta2';
A3 = [ones(m, 1), sigmoid(Z3)];

Z4 = A3 * Theta3';
A4 = sigmoid(Z4);

% Calculate cost
H = A4;
J = sum(sum(-Y .* log(H) - (1-Y) .* log(1 - H), 2)) / m;

J += sum(sum(Theta1(:,2:end) .^ 2, 2)) * lambda / m / 2;
J += sum(sum(Theta2(:,2:end) .^ 2, 2)) * lambda / m / 2;
J += sum(sum(Theta3(:,2:end) .^ 2, 2)) * lambda / m / 2;


% Backward prop

A4_grad = A4 - Y; % Ai_grad doesn't have bias term, but Ai does
A3_grad = (A4_grad * Theta3(:,2:end)) .* sigmoidGradient(Z3);
A2_grad = (A3_grad * Theta2(:,2:end)) .* sigmoidGradient(Z2);

delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));
delta_3 = zeros(size(Theta3));
for i = 1:m
	delta_1 += A2_grad(i, :)' * A1(i, :);
	delta_2 += A3_grad(i, :)' * A2(i, :);
	delta_3 += A4_grad(i, :)' * A3(i, :);
end
Theta1_grad = delta_1 ./ m;
Theta2_grad = delta_2 ./ m;
Theta3_grad = delta_3 ./ m;

% Backward prop with regulation
Theta1_grad(:, 2:end) += lambda .* Theta1(:, 2:end) ./ m;
Theta2_grad(:, 2:end) += lambda .* Theta2(:, 2:end) ./ m;
Theta3_grad(:, 2:end) += lambda .* Theta3(:, 2:end) ./ m;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
