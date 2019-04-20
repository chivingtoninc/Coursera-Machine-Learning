function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
i = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    % =============================================================

    predictions = (pval < epsilon);

    TP = predictions(predictions == 1 & yval == 1);
    FP = predictions(predictions == 1 & yval == 0);
    FN = predictions(predictions == 0 & yval == 1);

    nTP = size(TP,1);
    nFP = size(FP,1);
    nFN = size(FN,1);

    warning('off');
    pre = nTP / (nTP + nFP);
    rec = nTP / (nTP + nFN);
    F1 = (2 * pre * rec) / (pre + rec);

    % if i == 0 || mod(i, 100) == 0
    %   fprintf('\n');
    %   fprintf('Iter: %d\n', i++);
    %   fprintf('PV: %d x %d\n', size(pval));
    %   fprintf('YV: %d x %d\n', size(yval));
    %   fprintf('TP: %d\n', nTP);
    %   fprintf('FP: %d\n', nFP);
    %   fprintf('FN: %d\n', nFN);
    %   fprintf('pre: %d\n', pre);
    %   fprintf('rec: %d\n', rec);
    %   fprintf('F1: %d\n', F1);
    %   fprintf('\n\n');
    % end

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end
end
