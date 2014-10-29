clear
clc
close all
home
addpath ../util;

alphas = [0.08, 0.32, 1.28, 5.12];
batchSizes = [10, 25, 40];
epochNums = [40, 80];

for a = 1:4
    for b = 1:3
        for e = 1:2
            alpha = alphas(a);
            batchSize = batchSizes(b);
            numEpochs = epochNums(e);
            input_do_rate = 1;
            hidden_do_rate = 1;
            tic;
            er = test_nn(alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate);
            toc;
            disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
                alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate, er));

            input_do_rate = 1.0;
            hidden_do_rate = 0.9;
            tic;
            er = test_nn(alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate);
            toc;
            disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
                alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate, er));

            input_do_rate = 0.8;
            hidden_do_rate = 0.5;
            tic;
            er = test_nn(alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate);
            toc;
            disp(sprintf('alpha: %d batchSize: %d numEpochs: %d input_do_rate: %d hidden_do_rate: %d error: %d',...
                alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate, er));
        end
    end
end