%% Initialization
clear ; close all; clc; warning off;

%% Setup the parameters you will use for this exercise
input_layer_size  = 28 * 28;  % 28x28 Input Images of Digits
hidden_layer1_size = 300;
hidden_layer2_size = 100;
num_labels = 10;          % 10 labels, from 0 to 9
lambda = 0.5;

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

X = loadMNISTImages('data/train-images-idx3-ubyte')';
y = loadMNISTLabels('data/train-labels.idx1-ubyte');
X = X(1:10000, :);
y = y(1:10000, :);
m = size(X, 1);
% Trasform Y from 0~9 to logical vector
Y = zeros(size(y,1), num_labels);
for i = 1:size(Y,1)
  Y(i, y(i)+1) = 1;
end

% Loading Test Data
testX = loadMNISTImages('data/t10k-images.idx3-ubyte')';
testy = loadMNISTLabels('data/t10k-labels.idx1-ubyte');
testY = zeros(size(testy,1), num_labels);
for i = 1:size(testY,1)
	testY(i, testy(i)+1) = 1;
end

if 0
% Randomly select 100 datas to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;
end

fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

if 0
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nChecking gradient and initial_cost.\n');
% checkNNGradients;
initial_cost = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer1_size, hidden_layer2_size, num_labels, X, Y, lambda)
fprintf('Program paused. Press enter to continue.\n');
pause;
end

fprintf('\nTraining Neural Network normal gradient decend... \n');
total_iteration = 1500;
nn_params = initial_nn_params;
learning_rate = 0.5;

costs = zeros(1, total_iteration);
acc_trains = zeros(1, total_iteration);
acc_tests = zeros(1, total_iteration);
for iter = 1:total_iteration
	[cost, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer1_size, hidden_layer2_size, num_labels, X, Y, lambda);
	fprintf('iteration: %3d, cost: %f, ', iter, cost);
	nn_params -= learning_rate .* grad;
	costs(iter) = cost;

	[Theta1, Theta2, Theta3] = restoreTheta(nn_params, input_layer_size, hidden_layer1_size, hidden_layer2_size, num_labels);
	pred_train = predict(Theta1, Theta2, Theta3, X);
	pred_test = predict(Theta1, Theta2, Theta3, testX);
	acc_trains(iter) = mean(double(pred_train == (y+1))) * 100;
	acc_tests(iter) = mean(double(pred_test == (testy+1))) * 100;
	fprintf('accuracy(train, test): %f, %f\n', acc_trains(iter), acc_tests(iter));

	fflush(stdout);
end

save('paras.mat', 'input_layer_size', 'hidden_layer1_size', 'hidden_layer2_size', ...
	 'num_labels', 'nn_params', 'costs', 'total_iteration', 'lambda', 'learning_rate', ...
	 'acc_trains', 'acc_tests');

plot(1:total_iteration, costs);
plot(1:total_iteration, acc_trains);
plot(1:total_iteration, acc_tests);

fprintf('Program paused. Press enter to continue.\n');
pause;


[Theta1, Theta2, Theta3] = restoreTheta(nn_params, input_layer_size, hidden_layer1_size, hidden_layer2_size, num_labels);
pred_train = predict(Theta1, Theta2, Theta3, X);
pred_test = predict(Theta1, Theta2, Theta3, testX);
accuracy_train = mean(double(pred_train == (y+1))) * 100;
accuracy_test = mean(double(pred_test == (testy+1))) * 100;
fprintf('accuracy(train, test): %f, %f\n', accuracy_train, accuracy_test);