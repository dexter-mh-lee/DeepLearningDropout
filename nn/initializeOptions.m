function opt = initializeOptions()
	% opt = {alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate, gaussian, dropout}
	opt.alpha = 1;
	opt.batchSize = 50;
	opt.numEpochs = 10;
	opt.input_do_rate = 0.8;
	opt.hidden_do_rate = 0.5;
	opt.do_again_rate = 0;
	opt.gaussian = false;
	opt.dropout = true;
	opt.adaptive = false;
    opt.noiseScale = 0.06; % Found emperically. Relative noise to dropout rate
end