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

    if opt.sobol
        net.sobol = sobolset(net.nodeRanges(size(net.nodeRanges,1)-1, 2));
        %rand('state',0)
    end
    if opt.halton
        net.halton = haltonset(net.nodeRanges(size(net.nodeRanges,1)-1, 2));
        %rand('state',0)
    end
    
    m = size(x, 2);
    numBatches = m / opt.batchSize;
    if rem(numBatches, 1) ~= 0
        warning('numbatches not integer');
    end
    numBatches = floor(numBatches);
    net.numBatches = numBatches;
    for i = 1 : opt.numEpochs
        %disp(['epoch ' num2str(i) '/' num2str(opt.numEpochs)]);
        net.epochIndex = i;
        kk = randperm(m);
        meanTrainingError = 0;
        for l = 1 : numBatches
            net.batchIndex = l;
            % each column is one training instance
            batch_x = x(:, kk((l - 1) * opt.batchSize + 1 : l * opt.batchSize));
            batch_y = y(:, kk((l - 1) * opt.batchSize + 1 : l * opt.batchSize));

            net = feedForward_nn(net, batch_x, opt, i);
            net = backPropagation_nn(net, batch_y, opt);
            meanTrainingError = meanTrainingError + net.L;
        end
        meanTrainingError = meanTrainingError / numBatches;
        
        if i == opt.numEpochs
            if strcmp(opt.testerror, 'last')
                [er, bad] = testerror(net, test_x, test_y);
                net.testErrors = er;
            end
            if strcmp(opt.testerror_dropout, 'last')
                [erd, badd] = testerror_dropout(net, test_x, test_y, opt.input_do_rate(i), opt.hidden_do_rate(i), 100);
                net.testErrorsDropout = erd;
            end
            if strcmp(opt.trainingerror, 'last')
                net.trainingErrors = meanTrainingError;
            end
        end
        if strcmp(opt.testerror, 'all')
            [er, bad] = testerror(net, test_x, test_y);
            net.testErrors(i) = er;
        end
        if strcmp(opt.testerror_dropout, 'all')
            [erd, badd] = testerror_dropout(net, test_x, test_y, opt.input_do_rate(i), opt.hidden_do_rate(i), 100);
            net.testErrorsDropout(i) = erd;
        end
        if strcmp(opt.trainingerror, 'all')
            net.trainingErrors(i) = meanTrainingError;
        end
    end
    
%      net.layers{2}.w = net.layers{2}.w * opt.input_do_rate;
%      for l = 3:length(net.layers)
%          net.layers{l}.w = net.layers{l}.w * opt.hidden_do_rate;
%      end
    
    
end