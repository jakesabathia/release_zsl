function [acc,acc2] = run_sun_split(index);

lambda1_array = [10.^(-3:0)];
lambda2_array = [10.^(-3:0)];


%Cut Validation set
dataset = ['data_SUN_' num2str(index)];
load(dataset);
allindex = randperm(length(unique(train_labels)));
val_index = allindex(1:72);
train_index = allindex(73:end);
temp_train_Y = train_Y(:,train_index);
temp_val_Y = train_Y(:,val_index);

temp_train_X = [];
temp_train_labels = [];
for i=1:length(train_index)
	temp_index = find(train_labels==train_index(i));
	temp_train_X = [temp_train_X;train_X(temp_index,:)];
	temp_train_labels= [temp_train_labels;repmat(i,length(temp_index),1)];
end

temp_val_X = [];
temp_val_labels = [];
for i=1:length(val_index)
	temp_index = find(train_labels==val_index(i));
	temp_val_X = [temp_val_X;train_X(temp_index,:)];
	temp_val_labels= [temp_val_labels;repmat(i,length(temp_index),1)];
end

train_X = temp_train_X;
val_X = temp_val_X;
train_labels = temp_train_labels;
val_labels = temp_val_labels;
train_Y = temp_train_Y;
val_Y = temp_val_Y;

[train_X, xval_mean, xval_variance, xval_max] = normalization(train_X);
val_X = normalization(val_X, xval_mean, xval_variance, xval_max);
train_X = train_X';
val_X = val_X';

%Set Data
Data.train_X = train_X;
Data.test_X = val_X;
Data.train_Y = train_Y;
Data.test_Y = val_Y;
Data.train_labels = train_labels;
Data.test_labels = val_labels;

%Cross Validation
temp.acc = 0;
for i=lambda1_array
	param.lambda1 = i;
	for j=lambda2_array
		param.lambda2 = j;
		disp(['Cross Validation...Split_' num2str(index) ', lambda1:' num2str(param.lambda1) ', lambda2: ' num2str(param.lambda2)]);
		[acc] = Tao_inductive(Data,param);
		disp(['Validation acc:' num2str(100*acc)]);
		disp('============');
		if acc>temp.acc
			temp.acc = acc;
			temp.lambda1 = param.lambda1;
			temp.lambda2 = param.lambda2;
		end		
	end
end
param = temp;
clear temp;

%Load Data and Preprocessing
dataset = ['data_SUN_' num2str(index)];
load(dataset);
[train_X, xval_mean, xval_variance, xval_max] = normalization(train_X);
test_X = normalization(test_X, xval_mean, xval_variance, xval_max);

%Set Data
Data.train_X = train_X';
Data.test_X = test_X';
Data.train_Y = train_Y;
Data.test_Y = test_Y;
Data.train_labels = train_labels;
Data.test_labels = test_labels;

%Test
disp('====Inductive====');
disp(['Split:' num2str(index) ', lambda1:' num2str(param.lambda1) ', lambda2: ' num2str(param.lambda2)]);
[acc] = Tao_inductive(Data,param);
disp(['ACC:' num2str(acc*100)]);
%error('ddd');
%param.time = 20;
%disp('====Transductive====');
%disp(['Split:' num2str(index) ', lambda1:' num2str(param.lambda1) ', lambda2: ' num2str(param.lambda2)]);
%[acc2] = Tao_transductive(Data,param);
%disp(['ACC:' num2str(acc2*100)]);
acc2 = 0;
