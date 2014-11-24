function [er, bad] = testerror_dropout(net, x, y, ido, hdo, numTestEpochs)
    %  feedforward
    
    A = arrayfun(@(i) feedForward_test_dropout_nn(net, x, ido, hdo), 1:numTestEpochs, 'UniformOutput', false);
    s = size(A{1,1});
    A = cell2mat(A);
    A = reshape(A, [s, numTestEpochs]);
    
    a = mean(A, 3);
    
    [~, est] = max(a);
    [~, real] = max(y);
    bad = find(est ~= real);

    er = numel(bad) / size(y, 2);
end