clear
clc
%close all
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

structure_1 = {
        struct('type', 'I') %input layer
        struct('type', 'F', 'n', 64) %fully connected layer
        struct('type', 'F', 'n', 16) %fully connected layer
        struct('type', 'F', 'n', 4) %fully connected layer
        struct('type', 'O') %output layer
    };


structure_2 = {
        struct('type', 'I') %input layer
        struct('type', 'F', 'n', 64) %fully connected layer
        struct('type', 'F', 'n', 16) %fully connected layer
        struct('type', 'F', 'n', 16) %fully connected layer
        struct('type', 'F', 'n', 16) %fully connected layer
        struct('type', 'F', 'n', 16) %fully connected layer
        struct('type', 'F', 'n', 4) %fully connected layer
        struct('type', 'O') %output layer
    };



colors = ['k','r','g','b','c','m','y'];
do_rates = 0.5:0.1:1.0
opt.dropout = true;
% 
% opt.layers = structure_1;
% for i = 1:length(do_rates)
%     d = do_rates(i);
%     opt.input_do_rate = d;
%     opt.hidden_do_rate = d;
%     [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
%     plot(testErrors, 'Color', colors(i));
%     hold 'on';
%     disp(sprintf('structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
%     opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
% end
% figure;
% 
% opt.layers = structure_2;
% for i = 1:length(do_rates)
%     d = do_rates(i);
%     opt.input_do_rate = d;
%     opt.hidden_do_rate = d;
%     [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
%     plot(testErrors, 'Color', colors(i));
%     hold 'on';
%     disp(sprintf('structure: 2 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
%     opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
% end
% figure;
% opt.layers = structure_1;
% 
% opt.alpha = 8;
% for i = 1:length(do_rates)
%     d = do_rates(i);
%     opt.input_do_rate = d;
%     opt.hidden_do_rate = d;
%     [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
%     plot(testErrors, 'Color', colors(i));
%     hold 'on';
%     disp(sprintf('structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
%     opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
% end
% figure;
% opt.alpha = 1;
% 
% opt.alpha = 0.3;
% for i = 1:length(do_rates)
%     d = do_rates(i);
%     opt.input_do_rate = d;
%     opt.hidden_do_rate = d;
%     [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
%     plot(testErrors, 'Color', colors(i));
%     hold 'on';
%     disp(sprintf('structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
%     opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
% end
% figure;
% opt.alpha = 1;
% 
% opt.batchSize = 50;
% for i = 1:length(do_rates)
%     d = do_rates(i);
%     opt.input_do_rate = d;
%     opt.hidden_do_rate = d;
%     [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
%     plot(testErrors, 'Color', colors(i));
%     hold 'on';
%     disp(sprintf('structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
%     opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
% end
% figure;
% opt.batchSize = 10;
% 
% opt.gaussian = true;
% opt.dropout = false;
% opt.noiseScale = 0.03;
% for i = 1:length(do_rates)
%     d = do_rates(i);
%     opt.input_do_rate = d;
%     opt.hidden_do_rate = d;
%     [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
%     plot(testErrors, 'Color', colors(i));
%     hold 'on';
%     disp(sprintf('Gaussian. scale: %d structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
%     opt.noiseScale, opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
% end
% figure;
% opt.gaussian = false;
% opt.dropout = true;
% opt.noiseScale = 0.06;
% 
% opt.adaptive = true;
% for i = 1:length(do_rates)
%     d = do_rates(i);
%     opt.input_do_rate = d;
%     opt.hidden_do_rate = d;
%     [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
%     plot(testErrors, 'Color', colors(i));
%     hold 'on';
%     disp(sprintf('Adaptive. %d structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
%     opt.noiseScale, opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
% end
% figure;
% opt.adaptive = false;

% opt.dropout = false;
% opt.dropconnect = true;
% for i = 1:length(do_rates)
%     d = do_rates(i);
%     opt.input_do_rate = d;
%     opt.hidden_do_rate = d;
%     [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
%     plot(testErrors, 'Color', colors(i));
%     hold 'on';
%     disp(sprintf('DropConnect. %d structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
%     opt.noiseScale, opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
% end
% figure;
% opt.dropout = true;
% opt.dropconnect = false;
% 
% opt.numEpochs = 201;
% 
% opt.input_do_rate = 0.5:0.0025:1.0;
% opt.hidden_do_rate = opt.input_do_rate;
% tic;
% errors_d1 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, errors_d1(opt.numEpochs)));
% 
% opt.input_do_rate = 0.5:0.0025:1.0;
% opt.input_do_rate = (((opt.input_do_rate-0.5)*2).^2)/2+0.5;
% opt.hidden_do_rate = opt.input_do_rate;
% tic;
% errors_d2 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, errors_d2(opt.numEpochs)));
% 
% opt.input_do_rate = 1.0:-0.0025:0.5;
% opt.input_do_rate = 1-(((opt.input_do_rate-0.5)*2).^2)/2;
% opt.hidden_do_rate = opt.input_do_rate;
% tic;
% errors_d3 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, errors_d3(opt.numEpochs)));
% 
% opt.input_do_rate = 1.0:-0.0025:0.5;
% opt.hidden_do_rate = opt.input_do_rate;
% tic;
% errors_d4 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, errors_d4(opt.numEpochs)));
% 
% opt.input_do_rate = 1.0:-0.0025:0.5;
% opt.input_do_rate = (((opt.input_do_rate-0.5)*2).^2)/2+0.5;
% opt.hidden_do_rate = opt.input_do_rate;
% tic;
% errors_d5 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, errors_d5(opt.numEpochs)));
% 
% opt.input_do_rate = 0.5:0.0025:1.0;
% opt.input_do_rate = 1-(((opt.input_do_rate-0.5)*2).^2)/2;
% opt.hidden_do_rate = opt.input_do_rate;
% tic;
% errors_d6 = test_nn(opt);
% toc;
% disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
% opt.alpha, opt.batchSize, opt.numEpochs, errors_d6(opt.numEpochs)));
% 
% plot(1:opt.numEpochs, errors_d1, 'k', ...
%      1:opt.numEpochs, errors_d2, 'r', ...
%      1:opt.numEpochs, errors_d3, 'g', ...
%      1:opt.numEpochs, errors_d4, 'b', ...
%      1:opt.numEpochs, errors_d5, 'c', ...
%      1:opt.numEpochs, errors_d6, 'm');
% legend('linear increase 0.5-1', ...
%        'concave up increase 0.5-1', ...
%        'concave down increase 0.5-1', ...
%        'linear decrease 1-0.5', ...
%        'concave up decrease 1-0.5', ...
%        'concave down decrease 1-0.5');
% hold on;
% figure;
% 
% opt.input_do_rate = 0.5:0.0025:1.0;
% e1 = opt.input_do_rate;
% opt.input_do_rate = 0.5:0.0025:1.0;
% e2 = (((opt.input_do_rate-0.5)*2).^2)/2+0.5;
% opt.input_do_rate = 1.0:-0.0025:0.5;
% e3 = 1-(((opt.input_do_rate-0.5)*2).^2)/2;
% opt.input_do_rate = 1.0:-0.0025:0.5;
% e4 = opt.input_do_rate;
% opt.input_do_rate = 1.0:-0.0025:0.5;
% e5 = (((opt.input_do_rate-0.5)*2).^2)/2+0.5;
% opt.input_do_rate = 0.5:0.0025:1.0;
% e6 = 1-(((opt.input_do_rate-0.5)*2).^2)/2;
% plot(1:opt.numEpochs, e1, 'k', ...
%      1:opt.numEpochs, e2, 'r', ...
%      1:opt.numEpochs, e3, 'g', ...
%      1:opt.numEpochs, e4, 'b', ...
%      1:opt.numEpochs, e5, 'c', ...
%      1:opt.numEpochs, e6, 'm');
% legend('linear increase 0.5-1', ...
%        'convex increase 0.5-1', ...
%        'concave increase 0.5-1', ...
%        'linear decrease 1-0.5', ...
%        'convex decrease 1-0.5', ...
%        'concave decrease 1-0.5');



datasets = {'../data/cnae.mat',
'../data/banknote.mat',
'../data/winequality.mat',
'../data/pageblocks.mat',
'../data/wallrobot.mat',
'../data/magic04.mat',
'../data/letterrecognition.mat',
'../data/shuttle.mat'
'../data/gisette.mat',
};


opt.testerror = 'all';
opt.trainingerror = 'all';

results1 = []

for ds = 1:9
    opt.dataset = datasets{ds};
    for i = 1:length(do_rates)
        tic
        d = do_rates(i);
        opt.input_do_rate = d;
        opt.hidden_do_rate = d;
        [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
        %plot(testErrors, 'Color', colors(i));
        %hold 'on';
        disp(sprintf('Dropout. structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
        opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
        %opt.noiseScale, opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors, opt.dataset));
        toc
        results1.dropout{ds}{i}.dataset = datasets{ds};
        results1.dropout{ds}{i}.rate = do_rates(i);
        results1.dropout{ds}{i}.train = trainingErrors;
        results1.dropout{ds}{i}.test = testErrors;
    end
    %figure;
end

opt.adaptive = true;
for ds = 1:9
    opt.dataset = datasets{ds};
    for i = 1:length(do_rates)
        tic
        d = do_rates(i);
        opt.input_do_rate = d;
        opt.hidden_do_rate = d;
        [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
        %plot(testErrors, 'Color', colors(i));
        %hold 'on';
        disp(sprintf('Adaptive. structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
        opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
        %opt.noiseScale, opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors, opt.dataset));
        toc
        results1.adaptive{ds}{i}.dataset = datasets{ds};
        results1.adaptive{ds}{i}.rate = do_rates(i);
        results1.adaptive{ds}{i}.train = trainingErrors;
        results1.adaptive{ds}{i}.test = testErrors;
    end
    %figure;
end
opt.adaptive = false;

% opt.halton = true;
% for ds = 1:9
%     opt.dataset = datasets{ds};
%     for i = 1:length(do_rates)
%         tic
%         d = do_rates(i);
%         opt.input_do_rate = d;
%         opt.hidden_do_rate = d;
%         [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
%         %plot(testErrors, 'Color', colors(i));
%         %hold 'on';
%         disp(sprintf('Halton. structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
%          opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
%         %opt.noiseScale, opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors, opt.dataset));
%         toc
%         results1.halton{ds}{i}.dataset = datasets{ds};
%         results1.halton{ds}{i}.rate = do_rates(i);
%         results1.halton{ds}{i}.train = trainingErrors;
%         results1.halton{ds}{i}.test = testErrors;
%     end
%     %figure;
% end
% opt.halton = false;
% 
% opt.sobol = true;
% for ds = 1:8
%     opt.dataset = datasets{ds};
%     for i = 1:length(do_rates)
%         tic
%         d = do_rates(i);
%         opt.input_do_rate = d;
%         opt.hidden_do_rate = d;
%         [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
%         %plot(testErrors, 'Color', colors(i));
%         %hold 'on';
%         disp(sprintf('Sobol. structure: 1 alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
%          opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
%         %opt.noiseScale, opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors, opt.dataset));
%         toc
%         results1.sobol{ds}{i}.dataset = datasets{ds};
%         results1.sobol{ds}{i}.rate = do_rates(i);
%         results1.sobol{ds}{i}.train = trainingErrors;
%         results1.sobol{ds}{i}.test = testErrors;
%     end
%     %figure;
% end
% opt.sobol = false;

save('results1');
