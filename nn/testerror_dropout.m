function [er, bad] = testerror_dropout(net, x, y, ido, hdo, numTestEpochs, do_regression)
    %  feedforward
    if ~exist('do_regression','var')
        do_regression=false;
    end
    if do_regression
        A = arrayfun(@(i) feedForward_test_dropout_nn_regression(net, x, ido, hdo), 1:numTestEpochs, 'UniformOutput', false);
    else
        A = arrayfun(@(i) feedForward_test_dropout_nn(net, x, ido, hdo), 1:numTestEpochs, 'UniformOutput', false);
    end
    s = size(A{1,1});
    A = cell2mat(A);
    A = reshape(A, [s, numTestEpochs]);
    
    a = mean(A, 3);
    

    
    if do_regression
      est = a;
      real = y;
      bad = [];
      er = mean((est-real).^2);
    else
      [~, est] = max(a,[],1);
      [~, real] = max(y,[],1);
      bad = find(est ~= real);
      er = numel(bad) / size(y, 2);
    end
end