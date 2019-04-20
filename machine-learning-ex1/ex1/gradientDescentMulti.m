function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  %  Update theta by taking "num_iters" gradient steps with learning rate alpha, to
  % convuerge on an optimum value.

  m = length(y);
  n = size(X,2);
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters

    for j = 1:n
      d = 0;
      for i = 1:m
        x = X(i,:)';
        h = theta' * x;
        e = h - y(i);
        f = e * x(j);
        d = d + f;
      end
      d = d / m;
      theta(j) = theta(j) - alpha * d;
    end

    % theta = ((X' * X) ^ -1) * (X' * y);
    J_history(iter) = computeCostMulti(X, y, theta);
  end % GD STEPS
end % GD FUNCTION
