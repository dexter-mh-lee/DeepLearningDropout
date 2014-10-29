function net = backPropagation(net, y, alpha)
	numLayers = length(net.layers);

	e = net.layers{numLayers}.a - y; % Total error
	net.L = 1/2* sum(e(:) .^ 2) / size(e, 2); % Mean-squared loss for future checking
	net.layers{numLayers}.d = e .* (net.layers{numLayers}.a .* (1 - net.layers{numLayers}.a));

	% Compute delta values for each layer
	for l = (numLayers - 1) : -1 : 1
		net.layers{l}.d = (net.layers{l + 1}.w' * net.layers{l + 1}.d) .* (net.layers{l}.a .* (1 - net.layers{l}.a));
	end

	% Perform gradient descent, no weights for max-pooling layer
	for l = 2 : numLayers
		net.layers{l}.b = net.layers{l}.b - alpha * sum(net.layers{l}.d,2) / size(net.layers{l}.d,2);
		net.layers{l}.w = net.layers{l}.w - alpha * net.layers{l}.d * net.layers{l - 1}.a' / size(net.layers{l}.d,2);
	end

end