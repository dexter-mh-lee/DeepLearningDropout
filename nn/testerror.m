function [er, bad] = testerror(net, x, y, do_regression)
    if ~exist('do_regression','var')
      do_regression=false;
    end
    %  feedforward
    if do_regression
    else
    end
    if do_regression
      net = feedForward_test_nn_regression(net, x, 1, 1);
      h = net.layers{end}.a;
      a = y;
      bad = [];
      er = mean((h-a).^2);
    else
      net = feedForward_test_nn(net, x, 1, 1);
      [~, h] = max(net.layers{end}.a, [], 1);
      [~, a] = max(y, [], 1);
      bad = find(h ~= a);
      er = numel(bad) / size(y, 2);
    end
end