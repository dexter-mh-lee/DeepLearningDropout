function net = train_nn(net, x, y, alpha, batchSize, numEpochs)
    input_do_rate = 0.8; %Percent of nodes kept from input
    hidden_do_rate = 0.5;
    
    m = size(x, 2);
    numBatches = m / batchSize;
    if rem(numBatches, 1) ~= 0
        error('numbatches not integer');
    end
    for i = 1 : numEpochs
        disp(['epoch ' num2str(i) '/' num2str(numEpochs)]);
        tic;
        kk = randperm(m);
        for l = 1 : numBatches
            % each column is one training instance
            batch_x = x(:, kk((l - 1) * batchSize + 1 : l * batchSize));
            batch_y = y(:, kk((l - 1) * batchSize + 1 : l * batchSize));

            net = feedForward_nn(net, batch_x, input_do_rate, hidden_do_rate);
            net = backPropagation_nn(net, batch_y, alpha);
        end
        toc;
    end
    
    
    % if strcmp(net.layers{2}.type, 'C')
    %     for i = 1 : numel(net.layers{2}.k)
    %         for j = 1 : numel(net.layers{2}.k{i})
    %             net.layers{2}.k{i}{j} = net.layers{2}.k{i}{j}*input_do_rate;
    %         end
    %     end
    % end
    % if strcmp(net.layers{2}.type, 'F')
    %     for i = 1 : numel(net.layers{2}.w)
    %         for j = 1 : numel(net.layers{2}.w{i})
    %             net.layers{2}.w{i}{j} = net.layers{2}.w{i}{j}*input_do_rate;
    %         end
    %     end
    % end
    % for l = 3 : numel(net.layers)
    %     if strcmp(net.layers{l}.type, 'C')
    %         for i = 1 : numel(net.layers{l}.k)
    %             for j = 1 : numel(net.layers{l}.k{i})
    %                 net.layers{l}.k{i}{j} = net.layers{l}.k{i}{j}*hidden_do_rate;
    %             end
    %         end
    %     end
    %     if strcmp(net.layers{l}.type, 'F')
    %         for i = 1 : numel(net.layers{l}.w)
    %             for j = 1 : numel(net.layers{l}.w{i})
    %                 net.layers{l}.w{i}{j} = net.layers{l}.w{i}{j}*hidden_do_rate;
    %             end
    %         end
    %     end
    %     if strcmp(net.layers{l}.type, 'O')
    %         for i = 1 : numel(net.layers{l}.w)
    %             net.layers{l}.w{i} = net.layers{l}.w{i}*hidden_do_rate;
    %         end
    %     end
    % end
    
end