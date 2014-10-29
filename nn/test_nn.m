function output = test_nn(alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate);
    %%
    %load ../data/letterrecognition.mat
    %load ../data/mnist_uint8.mat
    load ../data/gisette.mat
    %% 

    rand('state',0)

    nn.layers = {
        struct('type', 'I') %input layer
        %struct('type', 'F', 'n', 1024) %fully connected layer
        %struct('type', 'F', 'n', 1024) %fully connected layer
        %struct('type', 'F', 'n', 256) %fully connected layer
        struct('type', 'F', 'n', 64) %fully connected layer
        struct('type', 'F', 'n', 16) %fully connected layer
        struct('type', 'F', 'n', 4) %fully connected layer
        struct('type', 'O') %output layer
    };

    train_x = reshape(train_x, size(train_x,1), size(train_x,3));
    test_x = reshape(test_x, size(test_x,1), size(test_x,3));
    % train_x = double(train_x');
    % test_x = double(test_x');
    % train_y = double(train_y');
    % test_y = double(test_y');
    %alpha = 0.3;
    %alpha = 1;
    %batchSize = 25;
    %numEpochs = 40;
    nn = setup_nn(nn, train_x, train_y);
    nn = train_nn(nn, train_x, train_y, alpha, batchSize, numEpochs, input_do_rate, hidden_do_rate);

    % testing

    [er, bad] = testerror(nn, test_x, test_y);

    output = er;
end

