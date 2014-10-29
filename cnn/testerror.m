function [er, bad] = testerror(net, x, y)
    %  feedforward
    net = feedForward(net, x, 1, 1);
    [~, h] = max(net.layers{end}.a{1});
    [~, a] = max(y);
    bad = find(h ~= a);

    er = numel(bad) / size(y, 2);