clear
clc
close all
home
addpath ../util;
%%
load ../data/letterrecognition.mat

%% 

rand('state',0)

nn.layers = {
    struct('type', 'I') %input layer
    struct('type', 'F', 'n', 16) %fully connected layer
    struct('type', 'F', 'n', 16) %fully connected layer
    struct('type', 'O') %output layer
};

train_x = reshape(train_x, size(train_x,1), size(train_x,3));
test_x = reshape(test_x, size(test_x,1), size(test_x,3));

alpha = 1;
batchSize = 50;
numEpochs = 5;
nn = setup_nn(nn, train_x, train_y);
nn = train_nn(nn, train_x, train_y, alpha, batchSize, numEpochs);

% testing

[er, bad] = testerror(nn, test_x, test_y);

fprintf('error = %d\n', er);

