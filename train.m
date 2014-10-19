function net = train(net, x, y, alpha, batchSize, numEpochs)
    m = size(x, 3);
    numBatches = m / batchSize;
    if rem(numBatches, 1) ~= 0
        error('numbatches not integer');
    end
    for i = 1 : numEpochs
        disp(['epoch ' num2str(i) '/' num2str(numEpochs)]);
        tic;
        kk = randperm(m);
        for l = 1 : numBatches
            batch_x = x(:, :, kk((l - 1) * batchSize + 1 : l * batchSize));
            batch_y = y(:,    kk((l - 1) * batchSize + 1 : l * batchSize));

            net = feedForward(net, batch_x);
            net = backPropagation(net, batch_y, alpha);
            % if isempty(net.rL)
            %     net.rL(1) = net.L;
            % end
            % net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end
end