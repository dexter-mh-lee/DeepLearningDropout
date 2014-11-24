function net = feedForward_nn(net, x, opt, epochNum)

	numLayers = length(net.layers); % total number of layers
	net.layers{1}.a = x;
  ido = opt.input_do_rate(epochNum);
  hdo = opt.hidden_do_rate(epochNum);
  if opt.gaussian
      noiseRate = 1-opt.noiseScale*(1-ido); %Scale noise from dropout rate.
      noiseSD = sqrt((1-noiseRate)/noiseRate); %Choose variance, then find standard deviation
      net.layers{1}.ga = normrnd(1, noiseSD, size(net.layers{1}.a));
      net.layers{1}.a = net.layers{1}.a .* net.layers{1}.ga;
  end
  net.layers{1}.dc = ones(size(net.layers{2}.w));
  if opt.dropout
      if opt.adaptive
          %We want 0s changed to 1s with probability 1 (dropped out
          %probability 0) and 1s changed to 0s with probability (1-ido)/ido
          threshold = 1 - net.layers{1}.do * (1-ido)/ido;
          %threshold = (1+ido)/2 - net.layers{1}.do * ((1+ido)/2 - 1 +((1-ido)/ido)*(1+ido)/2);
          net.layers{1}.do = rand(size(net.layers{1}.a)) <= threshold;
      else
          net.layers{1}.do = rand(size(net.layers{1}.a)) <= ido;
      end
      net.layers{1}.a = net.layers{1}.a .* net.layers{1}.do;
  elseif opt.dropconnect
      net.layers{1}.dc = rand(size(net.layers{2}.w)) <= ido;
  end
  net.layers{2}.wdc = net.layers{2}.w .* net.layers{1}.dc;

  for l = 2 : numLayers
    net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a, net.layers{l}.b));
    if l < numLayers && opt.gaussian
        noiseRate = 1-opt.noiseScale*(1-hdo);
        noiseSD = sqrt((1-noiseRate)/noiseRate);
        net.layers{l}.ga = normrnd(1, noiseSD, size(net.layers{l}.a));
        net.layers{l}.a = net.layers{l}.a .* net.layers{l}.ga;
    end
    if l < numLayers
      net.layers{l}.dc = ones(size(net.layers{l+1}.w));
      if opt.dropout
        if opt.adaptive && isFirst == false
            %As before, but with hdo
            threshold = 1 - net.layers{l}.do * (1-hdo)/hdo;
            %threshold = (1+hdo)/2 - net.layers{l}.do * ((1+hdo)/2 - 1 +((1-hdo)/hdo)*(1+hdo)/2);
            net.layers{l}.do = rand(size(net.layers{l}.a)) <= threshold;
        else
            net.layers{l}.do = rand(size(net.layers{l}.a)) <= hdo;
        end
        net.layers{l}.a = net.layers{l}.a .* net.layers{l}.do;
      elseif opt.dropconnect
        net.layers{l}.dc = rand(size(net.layers{l+1}.w)) <= hdo;
      end
      net.layers{l+1}.wdc = net.layers{l+1}.w .* net.layers{l}.dc;
    end
  end
end