function net = feedForward_nn(net, x, opt, epochNum)

	numLayers = length(net.layers); % total number of layers
	net.layers{1}.a = x;
    ido = opt.input_do_rate(epochNum);
    hdo = opt.hidden_do_rate(epochNum);
    if opt.boundedrandom
        ido = rand()*(1-ido)+ido;
        hdo = rand()*(1-hdo)+hdo;
    end
    if opt.gaussian
        noiseRate = 1-opt.noiseScale*(1-ido); %Scale noise from dropout rate.
        noiseSD = sqrt((1-noiseRate)/noiseRate); %Choose variance, then find standard deviation
        net.layers{1}.ga = normrnd(1, noiseSD, size(net.layers{1}.a));
        net.layers{1}.a = net.layers{1}.a .* net.layers{1}.ga;
    end
    if opt.dropout
        if opt.adaptive && epochNum > 1
            %We want 0s changed to 1s with probability 1 (dropped out
            %probability 0) and 1s changed to 0s with probability (1-ido)/ido
            threshold = 1 - net.layers{1}.do * (1-ido)/ido;
            %threshold = (1+ido)/2 - net.layers{1}.do * ((1+ido)/2 - 1 +((1-ido)/ido)*(1+ido)/2);
            net.layers{1}.do = repmat(rand(size(net.layers{1}.a,1),1),[1,opt.batchSize]) <= threshold;
        elseif opt.sobol
            do2 = repmat(rand(size(net.layers{1}.a,1),1),[1,opt.batchSize]) <= ido; %Not used, but makes results consistent
            batch = (net.epochIndex-1)*net.numBatches + net.batchIndex;
            batchSobol = net.sobol(batch, :);
            net.layers{1}.do = repmat((batchSobol(:,net.nodeRanges(1,1):net.nodeRanges(1,2)) <= ido)',[1,opt.batchSize]);
        elseif opt.halton
            do2 = repmat(rand(size(net.layers{1}.a,1),1),[1,opt.batchSize]) <= ido;
            batch = (net.epochIndex-1)*net.numBatches + net.batchIndex;
            batchHalton = net.halton(batch, :);
            net.layers{1}.do = repmat((batchHalton(:,net.nodeRanges(1,1):net.nodeRanges(1,2)) <= ido)',[1,opt.batchSize]);
        else
            net.layers{1}.do = repmat(rand(size(net.layers{1}.a,1),1),[1,opt.batchSize]) <= ido;
        end
        net.layers{1}.a = net.layers{1}.a .* net.layers{1}.do;
    elseif opt.dropconnect
        net.layers{1}.dc = rand(size(net.layers{2}.w)) <= ido;
        net.layers{2}.wdc = net.layers{2}.w .* net.layers{1}.dc;
    end

    for l = 2 : numLayers
        if opt.dropconnect
            net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.wdc * net.layers{l - 1}.a, net.layers{l}.b));
        else
            net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a, net.layers{l}.b));
        end
        if l < numLayers && opt.gaussian
            noiseRate = 1-opt.noiseScale*(1-hdo);
            noiseSD = sqrt((1-noiseRate)/noiseRate);
            net.layers{l}.ga = normrnd(1, noiseSD, size(net.layers{l}.a));
            net.layers{l}.a = net.layers{l}.a .* net.layers{l}.ga;
        end
        if l < numLayers
            if opt.dropout
                if opt.adaptive && epochNum > 1
                    %As before, but with hdo
                    threshold = 1 - net.layers{l}.do * (1-hdo)/hdo;
                    %threshold = (1+hdo)/2 - net.layers{l}.do * ((1+hdo)/2 - 1 +((1-hdo)/hdo)*(1+hdo)/2);
                    net.layers{l}.do = repmat(rand(size(net.layers{l}.a,1),1),[1,opt.batchSize]) <= threshold;
                elseif opt.sobol
                    do2 = repmat(rand(size(net.layers{l}.a,1),1),[1,opt.batchSize]) <= ido;
                    net.layers{l}.do = repmat((batchSobol(:,net.nodeRanges(l,1):net.nodeRanges(l,2)) <= hdo)',[1,opt.batchSize]);
                elseif opt.halton
                    do2 = repmat(rand(size(net.layers{l}.a,1),1),[1,opt.batchSize]) <= ido;
                    net.layers{l}.do = repmat((batchHalton(:,net.nodeRanges(l,1):net.nodeRanges(l,2)) <= hdo)',[1,opt.batchSize]);
                else
                    net.layers{l}.do = repmat(rand(size(net.layers{l}.a,1),1),[1,opt.batchSize]) <= hdo;
                end
                net.layers{l}.a = net.layers{l}.a .* net.layers{l}.do;
            elseif opt.dropconnect
                net.layers{l}.dc = rand(size(net.layers{l+1}.w)) <= hdo;
                net.layers{l+1}.wdc = net.layers{l+1}.w .* net.layers{l}.dc;
            end
        end
    end
end