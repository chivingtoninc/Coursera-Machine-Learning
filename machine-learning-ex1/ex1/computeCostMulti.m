function J = computeCostMulti(X, y, theta)
  m = length(y);
  J = 0;

  % ====================== YOUR CODE HERE ======================
  % Compute the cost of a particular choice of theta. You should set J to the cost.

  err_vec = (X * theta) - y;
  J = (err_vec' * err_vec) / (2 * m);

  % =========================================================================

end
