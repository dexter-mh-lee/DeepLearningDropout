clear
clc
close all
home
addpath ../util;

opt = initializeOptions();
opt.alpha = 1;
opt.batchSize = 10;
opt.numEpochs = 201;

opt.dropout = false;
opt.gaussian = false;

% opt.input_do_rate = 1;
% opt.hidden_do_rate = 1;
% tic;
% errors_o = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_o(opt.numEpochs)));
% 
opt.dropout = true;
opt.gaussian = false;
% 
% opt.input_do_rate = 0.75;
% opt.hidden_do_rate = 0.75;
% tic;
% errors_d1 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_d1(opt.numEpochs)));
% 
% opt.input_do_rate = 0.5;
% opt.hidden_do_rate = 0.5;
% tic;
% errors_d2 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_d2(opt.numEpochs)));


opt.adaptive = true;

opt.input_do_rate = 0.5:0.0025:1.0;
opt.hidden_do_rate = opt.input_do_rate;
tic;
errors_d1 = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, errors_d1(opt.numEpochs)));

opt.input_do_rate = 0.5:0.0025:1.0;
opt.input_do_rate = (((opt.input_do_rate-0.5)*2).^2)/2+0.5;
opt.hidden_do_rate = opt.input_do_rate;
tic;
errors_d2 = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, errors_d2(opt.numEpochs)));

opt.input_do_rate = 1.0:-0.0025:0.5;
opt.input_do_rate = 1-(((opt.input_do_rate-0.5)*2).^2)/2;
opt.hidden_do_rate = opt.input_do_rate;
tic;
errors_d3 = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, errors_d3(opt.numEpochs)));

opt.input_do_rate = 1.0:-0.0025:0.5;
opt.hidden_do_rate = opt.input_do_rate;
tic;
errors_d4 = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, errors_d4(opt.numEpochs)));

opt.input_do_rate = 1.0:-0.0025:0.5;
opt.input_do_rate = (((opt.input_do_rate-0.5)*2).^2)/2+0.5;
opt.hidden_do_rate = opt.input_do_rate;
tic;
errors_d5 = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, errors_d5(opt.numEpochs)));

opt.input_do_rate = 0.5:0.0025:1.0;
opt.input_do_rate = 1-(((opt.input_do_rate-0.5)*2).^2)/2;
opt.hidden_do_rate = opt.input_do_rate;
tic;
errors_d6 = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, errors_d6(opt.numEpochs)));

plot(1:opt.numEpochs, errors_d1, 'k', ...
     1:opt.numEpochs, errors_d2, 'r', ...
     1:opt.numEpochs, errors_d3, 'g', ...
     1:opt.numEpochs, errors_d4, 'b', ...
     1:opt.numEpochs, errors_d5, 'c', ...
     1:opt.numEpochs, errors_d6, 'm');
legend('linear increase 0.5-1', ...
       'concave up increase 0.5-1', ...
       'concave down increase 0.5-1', ...
       'linear decrease 1-0.5', ...
       'concave up decrease 1-0.5', ...
       'concave down decrease 1-0.5');
hold on;
figure;

opt.input_do_rate = 0.5:0.0025:1.0;
e1 = opt.input_do_rate;
opt.input_do_rate = 0.5:0.0025:1.0;
e2 = (((opt.input_do_rate-0.5)*2).^2)/2+0.5;
opt.input_do_rate = 1.0:-0.0025:0.5;
e3 = 1-(((opt.input_do_rate-0.5)*2).^2)/2;
opt.input_do_rate = 1.0:-0.0025:0.5;
e4 = opt.input_do_rate;
opt.input_do_rate = 1.0:-0.0025:0.5;
e5 = (((opt.input_do_rate-0.5)*2).^2)/2+0.5;
opt.input_do_rate = 0.5:0.0025:1.0;
e6 = 1-(((opt.input_do_rate-0.5)*2).^2)/2;
plot(1:opt.numEpochs, e1, 'k', ...
     1:opt.numEpochs, e2, 'r', ...
     1:opt.numEpochs, e3, 'g', ...
     1:opt.numEpochs, e4, 'b', ...
     1:opt.numEpochs, e5, 'c', ...
     1:opt.numEpochs, e6, 'm');
legend('linear increase 0.5-1', ...
       'convex increase 0.5-1', ...
       'concave increase 0.5-1', ...
       'linear decrease 1-0.5', ...
       'convex decrease 1-0.5', ...
       'concave decrease 1-0.5');

