function net = train_nn(net, x, y, test_x, test_y, opt)

    %Vectorize
    if length(opt.input_do_rate) == 1
        opt.input_do_rate = ones(opt.numEpochs, 1) * opt.input_do_rate;
    end
    if length(opt.hidden_do_rate) == 1
        opt.hidden_do_rate = ones(opt.numEpochs, 1) * opt.hidden_do_rate;
    end
    
    if length(opt.input_do_rate) ~= opt.numEpochs
        error('Invalid input dropout rate');
    end
    if length(opt.hidden_do_rate) ~= opt.numEpochs
        error('Invalid hidden dropout rate');
    end

    m = size(x, 2);
    numBatches = m / opt.batchSize;
    if rem(numBatches, 1) ~= 0
        error('numbatches not integer');
    end
    for i = 1 : opt.numEpochs
        %disp(['epoch ' num2str(i) '/' num2str(opt.numEpochs)]);
        kk = randperm(m);
        meanTrainingError = 0;
        for l = 1 : numBatches
            % each column is one training instance
            batch_x = x(:, kk((l - 1) * opt.batchSize + 1 : l * opt.batchSize));
            batch_y = y(:, kk((l - 1) * opt.batchSize + 1 : l * opt.batchSize));

            net = feedForward_nn(net, batch_x, opt, i);
            net = backPropagation_nn(net, batch_y, opt);
            meanTrainingError = meanTrainingError + net.L;
        end
        meanTrainingError = meanTrainingError / numBatches;
        [er, bad] = testerror(net, test_x, test_y);
        net.testErrors(i) = er;
        net.trainingErrors(i) = meanTrainingError;
    end
    
%      net.layers{2}.w = net.layers{2}.w * opt.input_do_rate;
%      for l = 3:length(net.layers)
%          net.layers{l}.w = net.layers{l}.w * opt.hidden_do_rate;
%      end
    
    
end