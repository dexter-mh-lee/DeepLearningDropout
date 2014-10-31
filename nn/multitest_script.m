clear
clc
close all
home
addpath ../util;

alphas = [1,8];
batchSizes = [10];
epochNums = [200];

for a = 1:2
    for b = 1:1
        for e = 1:1
            alpha = alphas(a);
            batchSize = batchSizes(b);
            numEpochs = epochNums(e);
            input_do_rate = 1;
            hidden_do_rate = 1;
            tic;
            errors_1 = test_nn(alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate);
            toc;
            disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
                alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate, errors_1(numEpochs)));

            input_do_rate = 1.0;
            hidden_do_rate = 0.9;
            tic;
            errors_2 = test_nn(alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate);
            toc;
            disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
                alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate, errors_2(numEpochs)));

            input_do_rate = 0.8;
            hidden_do_rate = 0.5;
            tic;
            errors_3 = test_nn(alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate);
            toc;
            disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
                alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate, errors_3(numEpochs)));
            
            plot(1:numEpochs, errors_1, 'r', 1:numEpochs, errors_2, 'g', 1:numEpochs, errors_3, 'b');
            hold on;
            figure;
        end
    end
end