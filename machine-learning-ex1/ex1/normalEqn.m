function [theta] = normalEqn(X, y)
  theta = zeros(size(X, 2), 1);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Complete the code to compute the closed form solution
  % to linear regression and put the result in theta.

  theta = ((X' * X) ^ -1) * (X' * y);

  % ============================================================
end
