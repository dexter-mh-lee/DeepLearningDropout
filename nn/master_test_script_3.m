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
opt.boundedrandom = false;
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
do_rates = 0.4:0.1:1.0
opt.dropout = true;

linear_increase = 0.5:0.5/199:1.0;
linear_decrease = 1.0:-0.5/199:0.5;
convex_decrease = linspace(-1, 0, opt.numEpochs).^6 * .5 + .5;
concave_decrease = linspace(0, 1, opt.numEpochs).^6 * -.5 + 1;
convex_increase = 1.5 - concave_decrease;
concave_increase = 1.5 - convex_decrease;
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

results = []
results.datasets = datasets;
results.do_rates = do_rates;

for ds = 1:length(datasets)
    opt.dataset = datasets{ds};
    tic
    d = linear_increase;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    disp(sprintf('linear increase error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    results.linear_increase{ds}.dataset = datasets{ds};
    results.linear_increase{ds}.train = trainingErrors;
    results.linear_increase{ds}.test = testErrors;
    tic
    d = linear_decrease;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    disp(sprintf('linear decrease error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    results.linear_decrease{ds}.dataset = datasets{ds};
    results.linear_decrease{ds}.train = trainingErrors;
    results.linear_decrease{ds}.test = testErrors;
    tic
    d = convex_increase;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    disp(sprintf('convex increase error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    results.convex_increase{ds}.dataset = datasets{ds};
    results.convex_increase{ds}.train = trainingErrors;
    results.convex_increase{ds}.test = testErrors;
    tic
    d = convex_decrease;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    disp(sprintf('convex decrease error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    results.convex_decrease{ds}.dataset = datasets{ds};
    results.convex_decrease{ds}.train = trainingErrors;
    results.convex_decrease{ds}.test = testErrors;
    tic
    d = concave_increase;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    disp(sprintf('concave increase error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    results.concave_increase{ds}.dataset = datasets{ds};
    results.concave_increase{ds}.train = trainingErrors;
    results.concave_increase{ds}.test = testErrors;
    tic
    d = concave_decrease;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    disp(sprintf('concave decrease error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    results.concave_decrease{ds}.dataset = datasets{ds};
    results.concave_decrease{ds}.train = trainingErrors;
    results.concave_decrease{ds}.test = testErrors;
    
    opt.boundedrandom = true;
    
    d = linear_increase_3;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    disp(sprintf('bounded random 0.3 error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    results.bounded_3{ds}.dataset = datasets{ds};
    results.bounded_3{ds}.train = trainingErrors;
    results.bounded_3{ds}.test = testErrors;
    
    d = linear_increase_7;
    opt.input_do_rate = d;
    opt.hidden_do_rate = d;
    [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
    disp(sprintf('bounded random 0.7 error: %d dataset: %s',...
    testErrors(opt.numEpochs), opt.dataset));
    toc
    results.bounded_7{ds}.dataset = datasets{ds};
    results.bounded_7{ds}.train = trainingErrors;
    results.bounded_7{ds}.test = testErrors;
    
    opt.boundedrandom = false;
    
    
end

save('results');

for ds = 1:length(datasets)
    opt.dataset = datasets{ds};
    for i = 1:length(do_rates)
        tic
        d = do_rates(i);
        opt.input_do_rate = d;
        opt.hidden_do_rate = d;
        [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
        disp(sprintf('Dropout. input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
        opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
        toc
        results.dropout{ds}{i}.dataset = datasets{ds};
        results.dropout{ds}{i}.rate = do_rates(i);
        results.dropout{ds}{i}.train = trainingErrors;
        results.dropout{ds}{i}.test = testErrors;
    end
end

save('results');

opt.adaptive = true;
for ds = 1:length(datasets)
    opt.dataset = datasets{ds};
    for i = 2:length(do_rates)
        tic
        d = do_rates(i);
        opt.input_do_rate = d;
        opt.hidden_do_rate = d;
        [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
        disp(sprintf('Adaptive. input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
        opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
        toc
        results.adaptive{ds}{i}.dataset = datasets{ds};
        results.adaptive{ds}{i}.rate = do_rates(i);
        results.adaptive{ds}{i}.train = trainingErrors;
        results.adaptive{ds}{i}.test = testErrors;
    end
end
opt.adaptive = false

save('results');

opt.halton = true;
for ds = 1:length(datasets)
    opt.dataset = datasets{ds};
    for i = 1:length(do_rates)
        tic
        d = do_rates(i);
        opt.input_do_rate = d;
        opt.hidden_do_rate = d;
        [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
        disp(sprintf('Halton. input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
        opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
        toc
        results.halton{ds}{i}.dataset = datasets{ds};
        results.halton{ds}{i}.rate = do_rates(i);
        results.halton{ds}{i}.train = trainingErrors;
        results.halton{ds}{i}.test = testErrors;
    end
end
opt.halton = false;

save('results');

opt.dropout = false;
opt.dropconnect = true;
for ds = 1:length(datasets)
    opt.dataset = datasets{ds};
    for i = 1:length(do_rates)
        tic
        d = do_rates(i);
        opt.input_do_rate = d;
        opt.hidden_do_rate = d;
        [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);
        disp(sprintf('dropconnect. input_do_rate: %d hidden_do_rate: %d error: %d dataset: %s',...
        opt.input_do_rate(1), opt.hidden_do_rate(1), testErrors(opt.numEpochs), opt.dataset));
        toc
        results.dropconnect{ds}{i}.dataset = datasets{ds};
        results.dropconnect{ds}{i}.rate = do_rates(i);
        results.dropconnect{ds}{i}.train = trainingErrors;
        results.dropconnect{ds}{i}.test = testErrors;
    end
end
opt.dropout = true;
opt.dropconnect = false;
save('results');
