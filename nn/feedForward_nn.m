function net = feedForward(net, x, input_do_rate, hidden_do_rate, dropout)

	if nargin < 5
        dropout = false;
    end


	numLayers = length(net.layers); % total number of layers
	net.layers{1}.a = x;
    if dropout
        net.layers{1}.do = rand(size(net.layers{1}.a)) <= input_do_rate;
        net.layers{1}.a = net.layers{1}.a .* net.layers{1}.do;
    end
	for l = 2:numLayers
		net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a, net.layers{l}.b));
        if dropout
            net.layers{l}.do = rand(size(net.layers{l}.a)) <= hidden_do_rate;
            net.layers{l}.a = net.layers{l}.a .* net.layers{l}.do;
        end
	end
end