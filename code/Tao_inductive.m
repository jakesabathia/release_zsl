function [acc,As,At,label] = Tao_inductive(data, param)

	%Load data
	X = data.train_X;
	S = data.train_Y;
	test_label = data.test_labels;
	test = data.test_X;
	test_Y = data.test_Y;
	X_label = data.train_labels;
 	lambda1 = param.lambda1;
 	lambda2 = param.lambda2;

 	%Constants
 	n_x = size(X,2);
 	n_s = size(S,2);

 	%Build semantic kernel matrix
	gamma = 1;
	S_k = kernel('rbf',S,S,gamma);
	test_k = kernel('rbf',S,test_Y,gamma);
	S = S_k;
	test_Y = test_k;
	clear S_k test_k;

	%Build H Matrix
	n = size(X,2) + size(S,2);
	H = eye(n) - 1/n*ones(n,n);

 	%Build M Matrix
 	M = sparse(zeros(n_x+n_s));
	for c = 1:size(S,2)
		e = sparse(zeros(n_x+n_s,1));
		e(find(X_label==c)) = 1/length(find(X_label==c));
		e(n_x+c) = -1;
		M = M + e*e';
	end

	%Build locality matrix
	L = sparse(zeros(n_x));
	for c = 1:size(S,2)
		gamma = 1;
		e = sparse(zeros(n_x,1));
		e(find(X_label==c)) = 1/length(find(X_label==c)); 
		L = L + e*e';
	end
	L = 2*(diag(sum(L,2)) - L);

	%Build blk Matrix
	T = blkdiag(X,S);
	L = blkdiag(L,zeros(size(S,2)));

	%Compute Eigen
	G = eye(size(X,1)+size(S,1));
	n = length(unique(X_label));
	[A,~] = eigs(T*(M+lambda2*L)*T'+lambda1*G,T*H*T',size(S,1),'SM');

	%Test
	As = A(1:size(X,1),:);
	At = A(size(X,1)+1:end,:);
	new_test_X = As'*test;
	new_test_Y = At'*test_Y;
	new_test_X = new_test_X*diag(1./sqrt(sum(new_test_X.^2)));
	new_test_Y = new_test_Y*diag(1./sqrt(sum(new_test_Y.^2)));
	[~,label] = max(new_test_Y'*new_test_X);
	acc = sum(label'==test_label)/length(test_label);