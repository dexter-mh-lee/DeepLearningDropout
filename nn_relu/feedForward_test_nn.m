function net = feedForward_test_nn(net, x, input_do_rate, hidden_do_rate, dropout)
% Instead of dropping out nodes, each weight is multiplied by the dropout
% rate. To be used on test data.
	if nargin < 5
        dropout = false;
  end


	numLayers = length(net.layers); % total number of layers
	net.layers{1}.a = x;
	for l = 2:numLayers
        if dropout
            if l == 2
                net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a * input_do_rate, net.layers{l}.b));
            else
                net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a * hidden_do_rate, net.layers{l}.b));
            end
        else
            net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a, net.layers{l}.b));
        end
	end
end