function net = setup(net, x , y)
	inputMaps = 1; % Number of input maps (input layer has one feature map - one image)
	mapSize = size(squeeze(x(:,:,1))); % Size of each feature map (size of input feature map = size of image)

	for l = 1:length(net.layers)
		if strcmp(net.layers{l}.type,'C') % Convolutional layer - initialize bias and kernel matrix
            assert(mapSize(1) >= net.layers{l}.scale, ['Layer ' num2str(l) ' size must be positive. Actual: ' num2str(mapSize)]);
			mapSize = mapSize - net.layers{l}.scale + 1; % patches overlap
            fan_out = net.layers{l}.outputMaps * net.layers{l}.scale ^ 2;
			for i = 1:net.layers{l}.outputMaps
                fan_in = inputMaps * net.layers{l}.scale ^ 2;
				for j = 1:inputMaps
					net.layers{l}.k{j}{i} = (rand(net.layers{l}.scale) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out)); % Initialize kernel - need to experiment with different methods
				end
            	net.layers{l}.b{i} = 0; % Initialize bias - need to experiment with different methods
			end
			inputMaps = net.layers{l}.outputMaps;

		elseif strcmp(net.layers{l}.type,'MP') % Max-pooling layer - initialize bias
			mapSize = mapSize / net.layers{l}.scale; % patches do not overlap
            assert(all(floor(mapSize)==mapSize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapSize)]);
            for j = 1:inputMaps
            	net.layers{l}.b{j} = 0; % Initialize bias - need to experiment with different methods
            end

		elseif strcmp(net.layers{l}.type,'F') % Fully connected layer
			mapSize = [prod(mapSize) 1]; % convert matrix to vector, but with same size
			for j = 1:inputMaps
				net.layers{l}.w{j} = rand(mapSize(1)) - 0.5; % Initialize weight matrix - need to experiment with different methods
				net.layers{l}.b{j} = 0; % Initialize bias - need to experiment with different methods
			end

		elseif strcmp(net.layers{l}.type,'O') % Output layer
            assert(l==length(net.layers), ['Output layer must be the final layer. Actual: ' num2str(mapSize)]);
			prev_mapSize = prod(mapSize) * inputMaps;
			mapSize = [size(y, 1) 1]; % if output layer, mapSize is the dimension of output
			net.layers{l}.w{1} = (rand(mapSize(1),prev_mapSize) - 0.5) * 2 * sqrt(6 / (mapSize(1) + prev_mapSize)); % Initialize weight matrix - need to experiment with different methods
			net.layers{l}.b{1} = zeros(mapSize(1),1);
		end
	end
end