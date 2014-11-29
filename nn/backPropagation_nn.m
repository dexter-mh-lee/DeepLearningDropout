function net = backPropagation_nn(net, y, opt)
	numLayers = length(net.layers);

	e = net.layers{numLayers}.a - y; % Total error
	net.L = 1/2* sum(e(:) .^ 2) / size(e, 2); % Mean-squared loss for future checking
	if opt.regression
    net.layers{numLayers}.d = e;
  else
    net.layers{numLayers}.d = e .* (net.layers{numLayers}.a .* (1 - net.layers{numLayers}.a));
  end
  
	% Compute delta values for each layer
	for l = (numLayers - 1) : -1 : 1
		if opt.gaussian
			grad = net.layers{l}.ga .* (net.layers{l}.a .* (1 - net.layers{l}.a));
		else
			grad = (net.layers{l}.a .* (1 - net.layers{l}.a));
    end	
    if opt.dropconnect
      net.layers{l}.d = (net.layers{l + 1}.wdc' * net.layers{l + 1}.d) .* grad;
    else
      net.layers{l}.d = (net.layers{l + 1}.w' * net.layers{l + 1}.d) .* grad;
    end
	end

	% Perform gradient descent, no weights for max-pooling layer
	for l = 2 : numLayers
		net.layers{l}.b = net.layers{l}.b - opt.alpha * sum(net.layers{l}.d,2) / size(net.layers{l}.d,2);
		net.layers{l}.w = net.layers{l}.w - opt.alpha * net.layers{l}.d * net.layers{l - 1}.a' / size(net.layers{l}.d,2);
    if opt.dropconnect
      net.layers{l}.wdc = net.layers{l}.w .* net.layers{l-1}.dc;
    end
	end

end