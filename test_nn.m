clear
close all
home
addpath util;
%%
load data/letterrecognition.mat

%% 

rand('state',0)

nn.layers = {
    struct('type', 'I') %input layer
    struct('type', 'F') %fully connected layer
    struct('type', 'O') %output layer
};

alpha = 1;
batchSize = 50;
numEpochs = 5;
nn = setup_cnn(nn, train_x, train_y);
nn = train_cnn(nn, train_x, train_y, alpha, batchSize, numEpochs);

%% testing

[er, bad] = testerror(nn, test_x, test_y);

fprintf('error = %d\n', er);

