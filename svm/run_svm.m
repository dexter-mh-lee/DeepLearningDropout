clear
close
home
%%
addpath('../../libsvm');
%%
datasets = {'gisette' 'letterrecognition' 'magic04' 'shuttle'};
%%
for d=datasets
  data = d{1};
  disp(data);
  load(['../data/' data]);
  [d, ~, nTrain] = size(train_x);
  nTest = size(test_x, 3);
  train_x = reshape(train_x, d, nTrain)';
  test_x = reshape(test_x, d, nTest)';
  %%
  nLabels = size(train_y, 1);
  train_y_old = train_y;
  train_y = nan(nTrain,1);
  for i=1:nLabels
    train_y(logical(train_y_old(i,:))') = i;
  end

  test_y_old = test_y;
  test_y = nan(nTest,1);
  for i=1:nLabels
    test_y(logical(test_y_old(i,:))') = i;
  end
  %%
  disp('rbf')
  tic
  rbf_model = svmtrain(train_y, train_x, '-t 3');
  toc
  [rbf_pred, rbf_acc, ~] = svmpredict(test_y, test_x, rbf_model);
  rbf_acc(1)
end