function net = backPropagation(net, y, alpha)
	numLayers = length(net.layers);

	e = net.layers{numLayers}.a{1} - y; % Total error
	net.L = 1/2* sum(e(:) .^ 2) / size(e, 2); % Mean-squared loss for future checking
	net.layers{numLayers}.d{1} = e .* (net.layers{numLayers}.a{1} .* (1 - net.layers{numLayers}.a{1}));

	inputLength = 1; % Size of the node in the second last layer
	for i = 1 : ndims(net.layers{numLayers - 1}.a{1}) - 1
		inputLength = inputLength * size(net.layers{numLayers - 1}.a{1},i);
	end
	% decompose the weight matrix into different input layers
	for j = 1 : length(net.layers{numLayers - 1}.a) 
		w{j} = net.layers{numLayers}.w{1}(:, (j - 1) * inputLength + 1: j * inputLength);
	end
	% Compute delta values for each layer
	for l = (numLayers - 1) : -1 : 1
		d = net.layers{l + 1}.d;
		if strcmp(net.layers{l}.type, 'C') % Convolutional layer - each node mapped to one node in the next layer
			for j = 1 : length(net.layers{l}.a)
				net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) ...
					.* (expand(d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) ...
					.* net.layers{l + 1}.k{j}); % Only update the delta values of nodes with maximum values
			end

		elseif strcmp(net.layers{l}.type, 'MP') % Max-pooling layer
			for i = 1:length(net.layers{l}.a)
				z = zeros(size(net.layers{l}.a{1})); 
				if strcmp(net.layers{l+1}.type, 'C') % Convolutional layer
					for j = 1:length(net.layers{l+1}.a)
						z = z + convn(d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
					end
					net.layers{l}.d{i} = z;
				else % Fully connected layer
					for j = 1:length(net.layers{l+1}.a)
						wd = w{i}' * d{j};
						z = z + reshape(wd, size(z)); % Change vector back to matrix
					end
					net.layers{l}.d{i} = z .* (net.layers{l}.a{i} .* (1 - net.layers{l}.a{i}));
				end
			end

		elseif strcmp(net.layers{l}.type, 'F')
			for i = 1:length(net.layers{l}.a)
				z = zeros(size(net.layers{l}.a{1}));
				for j = 1:length(net.layers{l + 1}.a) % Fully connected
					wd = w{i}' * d{j};
					z = z + wd;
				end
				net.layers{l}.d{i} = z .* (net.layers{l}.a{i} .* (1 - net.layers{l}.a{i}));
			end
			w = net.layers{l}.w;
		end
	end

	% Perform gradient descent, no weights for max-pooling layer
	for l = 2 : numLayers
		if strcmp(net.layers{l}.type, 'C') % Convolutional layer
			for j = 1:length(net.layers{l}.a)
				for i = 1:length(net.layers{l - 1}.a)
					dk = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
					net.layers{l}.k{i}{j} = net.layers{l}.k{i}{j} - alpha * dk; % Update k
				end
				db = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
				net.layers{l}.b{j} = net.layers{l}.b{j} - alpha * db; % Update bias
			end

		elseif strcmp(net.layers{l}.type, 'F') % Convolutional layer
			for j = 1:length(net.layers{l}.a)
				a = net.layers{l - 1}.a{j};
				dw = net.layers{l}.d{j} * reshape(a, size(a, 1) * size(a, 2), size(a, 3))' / size(net.layers{l}.d{j}, 3);
				net.layers{l}.w{j} = net.layers{l}.w{j} - alpha * dw; % Update weight matrix
				db = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
				net.layers{l}.b{j} = net.layers{l}.b{j} - alpha * db; % Update bias
			end

		elseif strcmp(net.layers{l}.type, 'O')
			% Combine result matrices from different input layers into one matrix
			a = [];
		    for j = 1 : length(net.layers{l - 1}.a)
		        sa = size(net.layers{l - 1}.a{j});
		        a = [a; reshape(net.layers{l - 1}.a{j}, inputLength, sa(ndims(net.layers{l - 1}.a{j})))];
		    end
		    dw = net.layers{l}.d{1} * a' /size(net.layers{l}.d{1}, 2);
		    net.layers{l}.w{1} = net.layers{l}.w{1} - alpha * dw;
		    db = mean(net.layers{l}.d{1}, 2);
		    net.layers{l}.b{1} = net.layers{l}.b{1} - alpha * db;
		end
	end

	function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end