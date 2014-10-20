function cnn = test
	load mnist_uint8;

	train_x = double(reshape(train_x',28,28,60000))/255;
	% test_x = double(reshape(test_x',28,28,10000))/255;
	train_y = double(train_y');
	% test_y = double(test_y');

	%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
	%will run 1 epoch in about 200 second and get around 11% error. 
	%With 100 epochs you'll get around 1.2% error

	rand('state',0)

	cnn.layers = {
	    struct('type', 'I') %input layer
	    struct('type', 'C', 'outputMaps', 6, 'scale', 5) %convolution layer
	    struct('type', 'MP', 'scale', 2) %sub sampling layer
	    struct('type', 'C', 'outputMaps', 12, 'scale', 5) %convolution layer
	    struct('type', 'MP', 'scale', 2) %subsampling layer
	    struct('type', 'O') %output layer
	};
	alpha = 1;
	batchSize = 50;
	numEpochs = 5;
	cnn = setup(cnn, train_x, train_y);
	cnn = train(cnn, train_x, train_y, alpha, batchSize, numEpochs);
end


% cnn = cnnsetup(cnn, train_x, train_y);
% cnn = cnntrain(cnn, train_x, train_y, opts);

% [er, bad] = cnntest(cnn, test_x, test_y);

% %plot mean squared error
% figure; plot(cnn.rL);
% assert(er<0.12, 'Too big error');
