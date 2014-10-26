function net = feedForward(net, x, input_do_rate, hidden_do_rate)
	numLayers = length(net.layers); % total number of layers
	net.layers{1}.a{1} = x;
	net.layers{1}.do{1} = rand(size(net.layers{1}.a{1})) <= input_do_rate;
  net.layers{1}.a{1} = net.layers{1}.a{1} .* net.layers{1}.do{1};
	inputMaps = 1; % Number of input feature maps
	% For each layer compute result matrices
	for l = 2:numLayers
		if strcmp(net.layers{l}.type, 'C')
			for j = 1:net.layers{l}.outputMaps
				% Weighted sum of each patch
				z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.scale - 1 net.layers{l}.scale - 1 0]);
				for i = 1:inputMaps
					z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
				end
				% final result of the layer is sigmoid of the sum plus the bias
				net.layers{l}.a{j} = sigmoid(z + net.layers{l}.b{j});
	            net.layers{l}.do{j} = rand(size(net.layers{l}.a{j})) <= hidden_do_rate; 
                net.layers{l}.a{j} = net.layers{l}.a{j} .* net.layers{l}.do{j};
			end
			% input feature maps for the next layer are the output maps for this layer
			inputMaps = net.layers{l}.outputMaps;

		elseif strcmp(net.layers{l}.type, 'MP')
			mapSize = size(net.layers{l - 1}.a{1}) / net.layers{l}.scale;
			for i = 1:inputMaps
				% apply max-pooling on each scale*scale patch
				[net.layers{l}.a{i}, net.layers{l}.k{i}] = maxPoolingLayer(net.layers{l - 1}.a{i}, mapSize, net.layers{l}.scale);
			end

		elseif strcmp(net.layers{l}.type, 'F')
			% Fully connected layer maps each input map to an output map
			for i = 1:inputMaps
				% If the previous layer was a convolutional or max-pooling layer convert to vectors
       			sa = size(net.layers{l - 1}.a{i});
				if(length(sa)>2) 
		          z = net.layers{l}.w{i} * reshape(net.layers{l - 1}.a{i}, sa(1) * sa(2), sa(3)); % If previous layer was convolutional or max-pooling layer convert matrix to vector
		        else
		          z = net.layers{l}.w{i} * net.layers{l - 1}.a{i};
				end
				
				% final result of the layer is sigmoid of the sum plus the bias
				net.layers{l}.a{i} = sigmoid(z + net.layers{l}.b{i});
	            net.layers{l}.do{i} = rand(size(net.layers{l}.a{i})) <= hidden_do_rate; 
                net.layers{l}.a{i} = net.layers{l}.a{i} .* net.layers{l}.do{i};
			end

		elseif strcmp(net.layers{l}.type, 'O')
			% Fully connected layer but the output dimension is the size of the label
			z = [];
			for j = 1 : inputMaps
				sa = size(net.layers{l - 1}.a{j});
				if(length(sa)>2) 
          z = [z;reshape(net.layers{l - 1}.a{j}, sa(1) * sa(2), sa(3))]; % If previous layer was convolutional or max-pooling layer convert matrix to vector
        else
          z = [z;net.layers{l - 1}.a{j}];
				end
			end
			net.layers{l}.a{1} = sigmoid(bsxfun(@plus, net.layers{l}.w{1} * z, net.layers{l}.b{1}));
				
		end
	end
end