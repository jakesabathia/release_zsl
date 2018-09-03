addpath('../../Utility');
clear;
buildpath;

%Set Dataset and Semantic Space
dataset = 'CUB'; %AWA, Dogs, CUB
semantictype = 'cont'; %cont, glove, word2vec, wordnet

%Cross Validation for Model Selection
load(['data_' dataset]);
[valtrain_X, xval_mean, xval_variance, xval_max] = normalization(valtrain_X);
val_X = normalization(val_X, xval_mean, xval_variance, xval_max);
Data.train_X = valtrain_X';
Data.test_X = val_X';
Data.train_Y = valtrain_Y(semantictype);
Data.test_Y = val_Y(semantictype);
Data.train_labels = valtrain_labels;
Data.test_labels = val_labels;

try
	load(['../param/' dataset '_' semantictype],'param');
catch
	temp.acc = 0;
	lambda1_array = 10.^(-3:0);
	lambda2_array = 10.^(-3:0);
	for i=lambda1_array
		for j=lambda2_array
			param.lambda1 = i;
			param.lambda2 = j;
			disp(['Cross Validation.. Dataset:' dataset ' ,Semantic:' semantictype ' ,lambda1:' num2str(param.lambda1) ' ,lambda2:' num2str(param.lambda2)]);
			[acc] = Tao_inductive(Data,param);
			disp(['Validation acc:' num2str(100*acc)]);
			disp('====');
			if acc>temp.acc
				temp.acc = acc;
				temp.lambda1 = param.lambda1;
				temp.lambda2 = param.lambda2;
			end
		end
	end
	param = temp;
	clear temp;
	save(['../param/' dataset '_' semantictype],'param');
end

%Test Inductive
[train_X, xtest_mean, xtest_variance, xtest_max] = normalization(train_X);
test_X = normalization(test_X, xtest_mean, xtest_variance, xtest_max);
Data.train_X = train_X';
Data.test_X = test_X';
Data.train_Y = train_Y(semantictype);
Data.test_Y = test_Y(semantictype);
Data.train_labels = train_labels;
Data.test_labels = test_labels;
disp('====Inductive====');
disp(['Dataset:' dataset ' ,Semantic:' semantictype ' ,lambda1:' num2str(param.lambda1) ' ,lambda2:' num2str(param.lambda2)]);
[acc] = Tao_inductive(Data,param);
disp(['Acc:' num2str(100*acc)]);
save(['../result/' dataset '_' semantictype '_induct'],'acc');

%Test Transductive
param.time = 10;
disp('====Transductive====');
disp(['Dataset:' dataset ' ,Semantic:' semantictype ' ,lambda1:' num2str(param.lambda1) ' ,lambda2:' num2str(param.lambda2)]);
[acc] = Tao_transductive(Data,param);
disp(['Acc:' num2str(100*acc)]);
save(['../result/' dataset '_' semantictype '_transduct'],'acc');

