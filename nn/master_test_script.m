clear
clc
close all
home
addpath ../util;

opt = initializeOptions();

opt.alpha = 1;
opt.batchSize = 10;
opt.numEpochs = 200;
opt.input_do_rate = 1.0;
opt.hidden_do_rate = 1.0;
opt.noiseScale = 0.06; % Found emperically. Relative noise to dropout rate
opt.gaussian = false;
opt.dropout = false;
opt.adaptive = false;
opt.dropconnect = false;
opt.dataset = '../data/gisette.mat';
opt.testerror = 'all';
opt.trainingerror = 'all';
opt.testerror_dropout = 'none';
opt.testerror_dropout_epochs = 100;

opt.layers = {
        struct('type', 'I') %input layer
        %struct('type', 'F', 'n', 1024) %fully connected layer
        %struct('type', 'F', 'n', 1024) %fully connected layer
        %struct('type', 'F', 'n', 256) %fully connected layer
        struct('type', 'F', 'n', 64) %fully connected layer
        struct('type', 'F', 'n', 16) %fully connected layer
        struct('type', 'F', 'n', 4) %fully connected layer
        struct('type', 'O') %output layer
    };

