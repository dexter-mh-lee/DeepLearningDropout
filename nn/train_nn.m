function net = train_nn(net, x, y, alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate)
    
    m = size(x, 2);
    numBatches = m / batchSize;
    if rem(numBatches, 1) ~= 0
        error('numbatches not integer');
    end
    for i = 1 : numEpochs
        %disp(['epoch ' num2str(i) '/' num2str(numEpochs)]);
        kk = randperm(m);
        for l = 1 : numBatches
            % each column is one training instance
            batch_x = x(:, kk((l - 1) * batchSize + 1 : l * batchSize));
            batch_y = y(:, kk((l - 1) * batchSize + 1 : l * batchSize));

            net = feedForward_nn(net, batch_x, input_do_rate, hidden_do_rate, true);
            net = backPropagation_nn(net, batch_y, alpha);
        end
    end
    
     net.layers{2}.w = net.layers{2}.w * input_do_rate;
     for l = 3:length(net.layers)
         net.layers{l}.w = net.layers{l}.w * hidden_do_rate;
     end
    
    
end