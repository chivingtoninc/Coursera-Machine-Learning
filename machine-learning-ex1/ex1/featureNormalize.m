function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));

  % ====================== YOUR CODE HERE ======================
  % Instructions: First, for each feature dimension, compute the mean
  % of the feature and subtract it from the dataset, storing the mean
  % value in mu. Next, compute the standard deviation of each feature
  % and divide each feature by it's standard deviation, storing the
  % standard deviation in sigma.

  mu = [mean(X(:,1)), mean(X(:,2))];
  sigma = [std(X(:,1)), std(X(:,2))];

  sizeNorm = (X(:,1) - mu(1)) / sigma(1);
  roomNorm = (X(:,2) - mu(2)) / sigma(2);
  X_norm = [sizeNorm, roomNorm];

  fprintf('\n\n');
  fprintf('mu:    '); disp(mu);
  fprintf('sigma: '); disp(sigma);
  fprintf('\n\n');

  % ============================================================

end % NORMALIZE FUNCTION
