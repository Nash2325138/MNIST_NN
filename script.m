%% Initialization
clear ; close all; clc; warning off;

%% Setup the parameters you will use for this exercise
input_layer_size  = 28 * 28;  % 28x28 Input Images of Digits
hidden_layer_size = 100;   % 25 hidden units
num_labels = 10;          % 10 labels, from 0 to 9

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

X = loadMNISTImages('data/train-images-idx3-ubyte')';
y = loadMNISTLabels('data/train-labels.idx1-ubyte');
m = size(X, 1);

% Trasform Y from 0~9 to logical vector
Y = zeros(size(y,1), num_labels);
for i = 1:size(Y,1)
  Y(i, y(i)+1) = 1;
end

% Randomly select 100 datas to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
fprintf('Program paused. Press enter to continue.\n');
pause;


fprintf('\nChecking gradient and initial_cost.\n');
% checkNNGradients;
initial_cost = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, 0)
fprintf('Program paused. Press enter to continue.\n');
pause;


fprintf('\nTraining Neural Network normal gradient decend... \n');
total_iteration = 300;
nn_params = initial_nn_params;
learning_rate = 0.04;
lambda = 1;
for iter = 1:total_iteration
  [cost, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda);
  fprintf('iteration: %-3d, cost: %f\n', iter, cost);
  fflush(stdout);
  nn_params -= learning_rate .* grad;
end

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


% fprintf('\nTraining Neural Network with fmincg... \n')
% options = optimset('MaxIter', 50);

% lambda = 1;
% costFunction = @(p) nnCostFunction(p, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size, ...
%                                    num_labels, X, y, lambda);
% [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                  hidden_layer_size, (input_layer_size + 1));

% Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                  num_labels, (hidden_layer_size + 1));

% fprintf('Program paused. Press enter to continue.\n');
% pause;



pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y+1)) * 100);
