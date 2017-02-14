function [Theta1, Theta2, Theta3] = restoreTheta(nn_params, input_layer_size, hidden_layer1_size, ...
												 hidden_layer2_size, num_labels)
% restore Theta 1~3 from unrolled nn_params
	end1 = hidden_layer1_size * (input_layer_size + 1);
	end2 = end1 + hidden_layer2_size * (hidden_layer1_size + 1);
	Theta1 = reshape(nn_params(1:end1), hidden_layer1_size, (input_layer_size + 1));
	Theta2 = reshape(nn_params((1 + end1):end2), hidden_layer2_size, (hidden_layer1_size + 1));
	Theta3 = reshape(nn_params((1 + end2):end), num_labels, (hidden_layer2_size + 1));
end