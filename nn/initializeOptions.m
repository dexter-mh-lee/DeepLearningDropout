function opt = initializeOptions()
	% opt = {alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate, gaussian, dropout}
	opt.alpha = 1;
	opt.batchSize = 50;
	opt.numEpochs = 10;
	opt.input_do_rate = 0.8;
	opt.hidden_do_rate = 0.5;
	opt.gaussian = false;
	opt.dropout = true;
	opt.adaptive = false;
    opt.sobol = false;
    opt.halton = false;
  opt.noiseScale = 0.06; % Found emperically. Relative noise to dropout rate
  opt.dropconnect = false;
  opt.boundedrandom = false;
  opt.dataset = '../data/gisette.mat';
    %../data/magic04.mat
    %../data/shuttle.mat
    %../data/letterrecognition.mat
    %../data/mnist_uint8.mat
  
  opt.testerror = 'all'; %Regular testing, 'all' to calculate on all epochs,
                         %'last' for last epoch only
  opt.trainingerror = 'all'; %Record training error
  opt.testerror_dropout = 'none'; %Testing with dropout and averaging
  opt.testerror_dropout_epochs = 100;
  
  opt.layers = {
        struct('type', 'I') %input layer
        %struct('type', 'F', 'n', 1024) %fully connected layer
        %struct('type', 'F', 'n', 1024) %fully connected layer
        %struct('type', 'F', 'n', 256) %fully connected layer
        struct('type', 'F', 'n', 64) %fully connected layer
        struct('type', 'F', 'n', 16) %fully connected layer
        struct('type', 'F', 'n', 4) %fully connected layer
        struct('type', 'O') %output layer
    };  
  
end