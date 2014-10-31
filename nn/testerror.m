function [er, bad] = testerror(net, x, y)
    %  feedforward
    net = feedForward_test_nn(net, x, 1, 1);
    [~, h] = max(net.layers{end}.a);
    [~, a] = max(y);
    bad = find(h ~= a);

    er = numel(bad) / size(y, 2);
end