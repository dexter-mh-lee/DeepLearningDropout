clear
clc
close all
home
addpath ../util;

alphas = [1,8];
batchSizes = [10];
epochNums = [200];

for a = 1:1%2
    for b = 1:1
        for e = 1:1
            opt = initializeOptions();
            opt.alpha = alphas(a);
            opt.batchSize = batchSizes(b);
            opt.numEpochs = epochNums(e);
            
            opt.input_do_rate = 1;
            opt.hidden_do_rate = 1;
            tic;
            errors_1 = test_nn(opt);
            toc;
            % er = output.er;
            disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
                opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_1(opt.numEpochs)));

            opt.input_do_rate = 1.0;
            opt.hidden_do_rate = 0.9;
            tic;
            errors_2 = test_nn(opt);
            toc;
            % er = output.er;
            disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
                opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_2(opt.numEpochs)));

            opt.input_do_rate = 0.8;
            opt.hidden_do_rate = 0.5;
            tic;
            errors_3 = test_nn(opt);
            toc;
            % er = output.er;
            disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
                opt.alpha, opt.batchSize, opt.numEpochs, opt.input_do_rate, opt.hidden_do_rate, errors_3(opt.numEpochs)));
            
            plot(1:opt.numEpochs, errors_1, 'r', 1:opt.numEpochs, errors_2, 'g', 1:opt.numEpochs, errors_3, 'b');
            hold on;
            figure;
        end
    end
end