clear
clc
close all
home
addpath ../util;

opt = initializeOptions();
opt.alpha = 8;
opt.batchSize = 10;
opt.numEpochs = 50;

% opt.dropout = false;
% opt.gaussian = false;

% opt.input_do_rate = 1;
% opt.hidden_do_rate = 1;
% tic;
% errors_o = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_o(opt.numEpochs)));

opt.dropout = true;

opt.input_do_rate = 0.5;
opt.hidden_do_rate = 0.5;
tic;
errors_d = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_d(opt.numEpochs)));

% opt.dropout = false;
% opt.gaussian = true;

% opt.input_do_rate = 0.995;
% opt.hidden_do_rate = 0.97;
% tic;
% errors_g2 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_g2(opt.numEpochs)));
% \

plot(1:opt.numEpochs, errors_d)