function [train_x, train_y, test_x, test_y] = split_train_test(X,y)
addpath ../util;

[n,d] = size(X);
nTrain = fix(n*2/3);
nTest = n-nTrain;
perm = randperm(n);
X = X(perm,:);
y = y(perm);
Xtrain = X(1:nTrain,:);
ytrain = y(1:nTrain);
Xtest = X(nTrain+1:n,:);
ytest = y(nTrain+1:n);

train_x = reshape(Xtrain', d, 1, nTrain);
train_y = makeY(ytrain);
test_x = reshape(Xtest', d, 1, nTest);
test_y = makeY(ytest);