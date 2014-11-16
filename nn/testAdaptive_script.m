clear
clc
close all
home
addpath ../util;

opt = initializeOptions();
opt.alpha = 8;
opt.batchSize = 10;
opt.numEpochs = 100;

% opt.dropout = false;
% opt.gaussian = false;

% opt.input_do_rate = 1;
% opt.hidden_do_rate = 1;
% tic;
% errors_o = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_o(opt.numEpochs)));


% opt.dropout = true;
% opt.gaussian = false;


% opt.input_do_rate = 0.5;
% opt.hidden_do_rate = 0.5;
% tic;
% errors_d = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_d(opt.numEpochs)));

opt.dropout = true;
opt.adaptive = true;


opt.input_do_rate = 0.5;
opt.hidden_do_rate = 0.5;
opt.do_again_rate = 0.3;
tic;
errors_a = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_a(opt.numEpochs)));

plot(1:opt.numEpochs, errors_o, 'k', ...
     1:opt.numEpochs, errors_d, 'b', ...
     1:opt.numEpochs, errors_a, 'r');
legend('normal input hidden', ...
       'dropout 0.5 0.5', ...
       'adaptive 0.5 0.5 0.3');
hold on;
figure;