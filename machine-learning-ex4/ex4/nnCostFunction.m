function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K = num_labels;

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

y_matrix = eye(num_labels)(y,:);

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

term1_matrix = y_matrix .* log(a3);
term2_matrix = (1 - y_matrix) .* log(1 - a3);
final_matrix = term1_matrix + term2_matrix;

sum = 0;

for i = 1:m
  for k = 1:K
    sum = sum + final_matrix(i,k);
  end
end

J = sum / (-m);

% -------------------------------------------------------------

Theta1_regTerm = 0;
for rows1 = 1:size(Theta1, 1)
  for cols1 = 2:size(Theta1, 2)
    Theta1_regTerm = Theta1_regTerm + (Theta1(rows1,cols1) ^ 2);
  end
end

Theta2_regTerm = 0;
for rows2 = 1:size(Theta2, 1)
  for cols2 = 2:size(Theta2, 2)
    Theta2_regTerm = Theta2_regTerm + (Theta2(rows2,cols2) ^ 2);
  end
end

J = J + ((Theta1_regTerm + Theta2_regTerm) * lambda) / (2 * m);

% -------------------------------------------------------------

d3 = a3 - y_matrix;
z2_grad = sigmoidGradient(z2);
d2 = (d3 * Theta2(:,2:end)) .* z2_grad;
Delta1 = d2' * a1;
Delta2 = d3' * a2;
Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Scaled_Theta1 = Theta1 * (lambda / m);
Scaled_Theta2 = Theta2 * (lambda / m);

Theta1_grad = Theta1_grad + Scaled_Theta1;
Theta2_grad = Theta2_grad + Scaled_Theta2;

% fprintf('\nd2 = (%dx%d)\n',   size(d2)); disp(d2);
% fprintf('\nd3 = (%dx%d)\n',   size(d3)); disp(d3);
% fprintf('\nDelta1 = (%dx%d)\n',    size(Delta1)); disp(Delta1);
% fprintf('\nDelta2 = (%dx%d)\n',     size(Delta2)); disp(Delta2);
% fprintf('\nz2 = (%dx%d)\n',   size(z2)); disp(z2);
% fprintf('\nsigmoidGradient(z2) = (%dx%d)\n',   size(sigmoidGradient(z2))); disp(sigmoidGradient(z2));
% fprintf('\na2 = (%dx%d)\n',   size(a2)); disp(a2);
% fprintf('\na3 = (%dx%d)\n',   size(a3)); disp(a3);

% fprintf('\nTheta1 = (%dx%d)\n',    size(Theta1)); disp(Theta1);
% fprintf('\nTheta1_grad  = (%dx%d)\n',    size(Theta1_grad)); disp(Theta1_grad);
% fprintf('\nTheta2 = (%dx%d)\n',     size(Theta2)); disp(Theta2);
% fprintf('\nTheta2_grad = (%dx%d)\n\n', size(Theta2_grad)); disp(Theta2_grad);

% fprintf('\n');
% fprintf('d3(%dx%d | 5000x10) = a3(%dx%d) - y_matrix(%dx%d)\n', size(d3), size(a3), size(y_matrix));
% fprintf('d2(%dx%d | 5000x25) = d3(%dx%d) * Theta2(%dx%d)\n', size(d2), size(d3), size(Theta2(:,2:end)));
% fprintf('Delta1(%dx%d | 25x400) = d2(%dx%d) * a1(%dx%d)\n', size(Delta1), size(d2'), size(a1(:,2:end)));
% fprintf('Delta2(%dx%d | 10x26) = d3(%dx%d) * a2(%dx%d)\n', size(Delta2), size(d3'), size(a2));
% fprintf('Theta1_grad(%dx%d | 25x400)\n', size(Theta1_grad));
% fprintf('Theta2_grad(%dx%d | 10x26)\n\n', size(Theta2_grad));

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
