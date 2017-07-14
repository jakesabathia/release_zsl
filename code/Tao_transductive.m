function [acc,As,At,label] = Tao_transductive(data, param)

	%Kernel Semantic Vector
	gamma = 1;
	Data.train_Y = kernel('rbf',[data.train_Y data.test_Y],data.train_Y,gamma);
	Data.test_Y = kernel('rbf',[data.train_Y data.test_Y],data.test_Y,gamma);
	
	%Iterative Solve
	Data.train_X = data.train_X;	
	Data.test_X = data.test_X;
	Data.train_labels = data.train_labels;
	Data.test_labels = data.test_labels;
	Data.As = [];
	Data.At = [];
	Data.psuedo_label = [];

	for time=1:param.time
		[acc,As,At,psuedolabel] = Tao_transductive_fun(Data, param);
		disp(['[' num2str(time) ']:' num2str(acc*100)]);
		Data.psuedo_label = psuedolabel;
		Data.As = As;
		Data.At = At;
	end
