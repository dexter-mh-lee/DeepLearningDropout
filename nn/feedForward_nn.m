function net = feedForward(net, x, input_do_rate, hidden_do_rate)
	numLayers = length(net.layers); % total number of layers
	net.layers{1}.a = x;
	for l = 2:numLayers
		net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a, net.layers{l}.b));
	end
end