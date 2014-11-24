function [testErrors, trainingErrors, testErrorsDropout] = test_nn(opt);

    %%
    load(opt.dataset);
    %% 

    rand('state',0)

    nn.layers = opt.layers;

    nn.testErrors = zeros(opt.numEpochs,1);
    nn.testErrorsDropout = zeros(opt.numEpochs,1);

    nn.trainingErrors = zeros(opt.numEpochs,1);

    train_x = reshape(train_x, size(train_x,1), size(train_x,3));
    test_x = reshape(test_x, size(test_x,1), size(test_x,3));
    
    nn = setup_nn(nn, train_x, train_y);
    nn = train_nn(nn, train_x, train_y, test_x, test_y, opt);
    
    if strcmp(opt.testerror, 'all') | strcmp(opt.testerror, 'last')
        testErrors = nn.testErrors;
    else
        testErrors = [];
    end
    if strcmp(opt.testerror_dropout, 'all') | strcmp(opt.testerror, 'last')
        testErrorsDropout = nn.testErrorsDropout;
    else
        testErrorsDropout = [];
    end
    if strcmp(opt.trainingerror, 'all') | strcmp(opt.trainingerror, 'last')
        trainingErrors = nn.trainingErrors;
    else
        trainingErrors = [];
    end

end

