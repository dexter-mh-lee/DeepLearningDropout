clear
clc
close all
home
addpath ../util;

numEpochs = 100;
opt = initializeOptions();
opt.alpha = 1; 
opt.batchSize = 10;
opt.numEpochs = numEpochs;

dropoutRates = 0.85;

pTesterrorTrainingerror = zeros(length(dropoutRates), numEpochs, 2);

for do = 1:length(dropoutRates)
    tic
    opt.input_do_rate = dropoutRates(do);
    opt.hidden_do_rate = dropoutRates(do);
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    plot(testErrors, 'Color', 'r');
    hold 'on';
    plot(testErrorsDropout, 'Color', 'b');
    pTesterrorTrainingerror(do,:,1) = reshape(testErrors, [1, numEpochs, 1]);
    pTesterrorTrainingerror(do,:,2) = reshape(trainingErrors, [1, numEpochs, 1]);
    toc
end

save('result.mat','pTesterrorTrainingerror');

% 
% alphas = [0.1, 0.3, 0.5, 1, 4, 8, 16];
% dropoutRates = [0.5, 0.65, 0.8, 0.95, 1.0];
% noiseRates = [0.95, 0.98, 0.995, 0.999, 1.0];
% batchSize = 10;
% maxEpoch = 200;
% 
% for a = 1:length(alphas)
%     opt.dropout = true;
%     opt.gaussian = false;
%     for i = 1:length(dropoutRates)
%         for h = 1:length(dropoutRates)
%             opt = initializeOptions();
%             opt.alpha = alphas(a);
%             opt.batchSize = batchSize;
%             opt.numEpochs = maxEpoch;
%             opt.input_do_rate = dropoutRates(i);
%             opt.hidden_do_rate = dropoutRates(h);
%             
%             tic;
%             errors = test_nn(opt);
%             toc;
%             disp(sprintf('dropout'));
%             disp(sprintf('alpha: %d input_do_rate: %d hidden_do_rate: %d',...
%                 opt.alpha, opt.input_do_rate, opt.hidden_do_rate));
%             [ time, value ] = findConvergence(errors);
%             disp(sprintf('final error: %d convergence time: %d', value, time))
%             
% %             plot(1:opt.numEpochs, errors_1, 'r', 1:opt.numEpochs, errors_2, 'g', 1:opt.numEpochs, errors_3, 'b');
% %             hold on;
% %             figure;
%         end
%     end
%     opt.dropout = false;
%     opt.gaussian = true;
%     for i = 1:length(noiseRates)
%         for h = 1:length(noiseRates)
%             opt = initializeOptions();
%             opt.alpha = alphas(a);
%             opt.batchSize = batchSize;
%             opt.numEpochs = maxEpoch;
%             opt.input_do_rate = noiseRates(i);
%             opt.hidden_do_rate = noiseRates(h);
%             
%             tic;
%             errors = test_nn(opt);
%             toc;
%             disp(sprintf('gaussian noise'));
%             disp(sprintf('alpha: %d input_do_rate: %d hidden_do_rate: %d',...
%                 opt.alpha, opt.input_do_rate, opt.hidden_do_rate));
%             [ time, value ] = findConvergence(errors);
%             disp(sprintf('final error: %d convergence time: %d', value, time))
%             
% %             plot(1:opt.numEpochs, errors_1, 'r', 1:opt.numEpochs, errors_2, 'g', 1:opt.numEpochs, errors_3, 'b');
% %             hold on;
% %             figure;
%         end
%     end
% end