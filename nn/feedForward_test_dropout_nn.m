function output = feedForward_test_dropout_nn(net, x, input_do_rate, hidden_do_rate)
    numLayers = length(net.layers); % total number of layers
    net.layers{1}.a = x;
    ido = input_do_rate;
    hdo = hidden_do_rate;
    
    net.layers{1}.do = rand(size(net.layers{1}.a)) <= ido;
    net.layers{1}.a = net.layers{1}.a .* net.layers{1}.do;

	for l = 2 : numLayers
		net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a, net.layers{l}.b));
        if l < numLayers
            net.layers{l}.do = rand(size(net.layers{l}.a)) <= hdo;
            net.layers{l}.a = net.layers{l}.a .* net.layers{l}.do;
        end
    end
    output = net.layers{end}.a;
end