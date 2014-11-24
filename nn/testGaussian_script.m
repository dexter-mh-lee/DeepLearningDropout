clear
clc
close all
home
addpath ../util;

opt = initializeOptions();
opt.alpha = 1;
opt.batchSize = 10;
opt.numEpochs = 100;

opt.dropout = false;
opt.gaussian = false;

opt.input_do_rate = 1;
opt.hidden_do_rate = 1;
tic;
errors_o = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_o(opt.numEpochs)));

opt.dropout = false;
opt.gaussian = true;

opt.input_do_rate = 0.5;
opt.hidden_do_rate = 0.92;
%opt.noiseScale = 1;
tic;
errors_g1 = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_g1(opt.numEpochs)));

opt.input_do_rate = 0.92;
opt.hidden_do_rate = 0.5;
tic;
errors_g2 = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_g2(opt.numEpochs)));

% opt.input_do_rate = 0.92;
% opt.hidden_do_rate = 0.5;
% tic;
% errors_g3 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_g3(opt.numEpochs)));
% 
% opt.input_do_rate = 0.5;
% opt.hidden_do_rate = 1;
% tic;
% errors_g4 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_g4(opt.numEpochs)));

% opt.dropout = true;
% opt.gaussian = false;
% 
% opt.input_do_rate = 1.0;
% opt.hidden_do_rate = 0.9;
% tic;
% errors_d1 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_d1(opt.numEpochs)));
% 
% opt.input_do_rate = 0.9;
% opt.hidden_do_rate = 1.0;
% tic;
% errors_d2 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_d2(opt.numEpochs)));
% 
% opt.input_do_rate = 0.8;
% opt.hidden_do_rate = 0.5;
% tic;
% errors_d3 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_d3(opt.numEpochs)));
% 
% opt.input_do_rate = 0.5;
% opt.hidden_do_rate = 0.8;
% tic;
% errors_d4 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_d4(opt.numEpochs)));

plot(1:opt.numEpochs, errors_o, 'k', ...
     1:opt.numEpochs, errors_g1, 'r', ...
     1:opt.numEpochs, errors_g2, 'g');%, ...
%      1:opt.numEpochs, errors_g3, 'b', ...
%      1:opt.numEpochs, errors_g4, 'm');%, ...
%      1:opt.numEpochs, errors_d1, 'b', ...
%      1:opt.numEpochs, errors_d2, 'c', ...
%      1:opt.numEpochs, errors_d3, 'm', ...
%      1:opt.numEpochs, errors_d4, 'y');
legend('input sd = 0.0, hidden sd = 0.0', ...
       'input sd = 0.0709, hidden sd = 0.1759', ...
       'input sd = 0.1759, hidden sd = 0.0709');%, ...
%        'input sd = 0.0709, hidden sd = 0.0709', ...
%        'input sd = 0.0709, hidden sd = 0.0');%, ...
%        'dropout 1.0 0.9', ...
%        'dropout 0.9 1.0', ...
%        'dropout 0.8 0.5', ...
%        'dropout 0.5 0.8');
hold on;
figure;