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

 
linear_increase = 0.5:0.5/199:1.0;
linear_decrease = 1.0:-0.5/199:0.5;

convex_increase = (linear_increase.*linear_increase - 0.25)*2/3 + 0.5;
convex_decrease = (linear_decrease.*linear_decrease - 0.25)*2/3 + 0.5;
concave_increase = 1.0 - convex_decrease;
concave_decrease = 1.0 - convex_increase;


linear_increase_3 = 0.3:0.7/199:1.0;
linear_increase_7 = 0.7:0.3/199:1.0;

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

resultsex = []

for ds = 1:9
    opt.dataset = datasets{ds};
    tic
    d = linear_increase;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    %plot(testErrors, 'Color', colors(i));
    %hold 'on';
    disp(sprintf('linear increase error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    resultsex.dropout{ds}{1}.dataset = datasets{ds};
    resultsex.dropout{ds}{1}.train = trainingErrors;
    resultsex.dropout{ds}{1}.test = testErrors;
    tic
    d = linear_decrease;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    %plot(testErrors, 'Color', colors(i));
    %hold 'on';
    disp(sprintf('linear decrease error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    resultsex.dropout{ds}{2}.dataset = datasets{ds};
    resultsex.dropout{ds}{2}.train = trainingErrors;
    resultsex.dropout{ds}{2}.test = testErrors;
    tic
    d = convex_increase;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    %plot(testErrors, 'Color', colors(i));
    %hold 'on';
    disp(sprintf('convex increase error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    resultsex.dropout{ds}{3}.dataset = datasets{ds};
    resultsex.dropout{ds}{3}.train = trainingErrors;
    resultsex.dropout{ds}{3}.test = testErrors;
    tic
    d = convex_decrease;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    %plot(testErrors, 'Color', colors(i));
    %hold 'on';
    disp(sprintf('convex decrease error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    resultsex.dropout{ds}{4}.dataset = datasets{ds};
    resultsex.dropout{ds}{4}.train = trainingErrors;
    resultsex.dropout{ds}{4}.test = testErrors;
    tic
    d = concave_increase;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    %plot(testErrors, 'Color', colors(i));
    %hold 'on';
    disp(sprintf('concave increase error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    resultsex.dropout{ds}{5}.dataset = datasets{ds};
    resultsex.dropout{ds}{5}.train = trainingErrors;
    resultsex.dropout{ds}{5}.test = testErrors;
    tic
    d = concave_decrease;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    %plot(testErrors, 'Color', colors(i));
    %hold 'on';
    disp(sprintf('concave decrease error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    resultsex.dropout{ds}{6}.dataset = datasets{ds};
    resultsex.dropout{ds}{6}.train = trainingErrors;
    resultsex.dropout{ds}{6}.test = testErrors;
    
    opt.boundedrandom = true;
    
    d = linear_increase_3;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    %plot(testErrors, 'Color', colors(i));
    %hold 'on';
    disp(sprintf('bounded random 0.3 error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    resultsex.dropout{ds}{7}.dataset = datasets{ds};
    resultsex.dropout{ds}{7}.train = trainingErrors;
    resultsex.dropout{ds}{7}.test = testErrors;
    
    d = linear_increase_7;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    %plot(testErrors, 'Color', colors(i));
    %hold 'on';
    disp(sprintf('bounded random 0.3 error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    resultsex.dropout{ds}{8}.dataset = datasets{ds};
    resultsex.dropout{ds}{8}.train = trainingErrors;
    resultsex.dropout{ds}{8}.test = testErrors;
    
    opt.boundedrandom = false;
end

opt.dropout = false;
opt.dropconnect = true;
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
        resultsex.dropconnect{ds}{i}.dataset = datasets{ds};
        resultsex.dropconnect{ds}{i}.rate = do_rates(i);
        resultsex.dropconnect{ds}{i}.train = trainingErrors;
        resultsex.dropconnect{ds}{i}.test = testErrors;
    end
    %figure;
end
opt.dropout = true;
opt.dropconnect = false;
opt.adaptive = false;
save('resultsex');
