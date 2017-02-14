X = loadMNISTImages('data/train-images-idx3-ubyte')';
y = loadMNISTLabels('data/train-labels.idx1-ubyte');
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

load('paras.mat')

[Theta1, Theta2, Theta3] = restoreTheta(nn_params, input_layer_size, hidden_layer1_size, hidden_layer2_size, num_labels);
pred_train = predict(Theta1, Theta2, Theta3, X);
pred_test = predict(Theta1, Theta2, Theta3, testX);
accuracy_train = mean(double(pred_train == (y+1))) * 100;
accuracy_test = mean(double(pred_test == (testy+1))) * 100;
fprintf('accuracy(train, test): %f, %f\n', accuracy_train, accuracy_test);