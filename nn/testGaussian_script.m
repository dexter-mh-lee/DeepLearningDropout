clear
clc
close all
home
addpath ../util;

opt = initializeOptions();
opt.alpha = 0.3;
opt.batchSize = 10;
opt.numEpochs = 10;

opt.input_do_rate = 1;
opt.hidden_do_rate = 1;

% tic;
% errors_o = test_nn(opt);
% toc;

% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_o(opt.numEpochs)));

opt.input_do_rate = 0.8;
opt.hidden_do_rate = 0.9;

opt.dropout = false;
opt.gaussian = true;

tic;
errors_g = test_nn(opt);
toc;

% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_g(opt.numEpochs)));

% plot(1:opt.numEpochs, errors_o, 'r', 1:opt.numEpochs, errors_g, 'b');
hold on;
% figure;