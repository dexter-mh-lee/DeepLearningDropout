clear
clc
close all
home
addpath ../util;

opt = initializeOptions();
opt.alpha = 1;
opt.batchSize = 1000;
opt.numEpochs = 1;

opt.dropout = false;
opt.gaussian = false;
opt.dropconnect = true;



opt.input_do_rate = 0.5;
opt.hidden_do_rate = 0.5;
tic;
errors_o = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_o(opt.numEpochs)));