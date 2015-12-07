function [error_train, error_val] = ...
    TrainingOnRandoms(X, y, Xval, yval, lambda)

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
RepCount = 50; % From the assignment

for i = 1:m
    for r = 1:RepCount
        ThisRepExamples = randsample(m,i);
        ThisX = X(ThisRepExamples,:);
        ThisY = y(ThisRepExamples,:);
        theta = trainLinearReg(ThisX, ThisY, lambda);
        [J, grad] = linearRegCostFunction(ThisX, ThisY, theta, 0);
        error_train(i) = error_train(i) + J;
        [J, grad] = linearRegCostFunction(Xval, yval, theta, 0);
        error_val(i) =  error_val(i) + J;
    end
end

error_train = error_train / RepCount;
error_val =  error_val / RepCount;

end % TrainingOnRandoms


