function [acc,As,At,label] = Tao_transductive_fun(data, param)

	%load variable
	X = data.train_X;
	S = data.train_Y;
	test_label = data.test_labels;
	test = data.test_X;
	test_Y = data.test_Y;
	X_label = data.train_labels;
	psuedo_label = data.psuedo_label;
	As = data.As;
	At = data.At;
 	lambda1 = param.lambda1;
 	lambda2 = param.lambda2;

 	%Initialize
 	if ~isequal(psuedo_label,[])
 		psuedo_label= psuedo_label+ size(S,2);
 	end
 	label_all = [X_label;psuedo_label];

 	N = cell(length(unique(label_all)),1);
 	if isequal(As,[])

 		S_all = S;
 		X_all = X;
 		for i = 1:size(S,2)
 			N{i} = 1/length(find(X_label==i))*ones(length(find(X_label==i)),1);
 		end
 	else
		S_all = [S test_Y];
		X_all = [X test];
 		gamma = 1;
 		for c = 1:length(N)
			N{c} = 1/length(find(label_all==c))*ones(length(find(label_all==c)),1);
		end
 	end

 	n = size(X,2) + size(test,2) + size(S_all,2);
 	H = eye(n) - 1/n*ones(n,n);

 	%M Matrix
 	M = sparse(zeros(size(X_all,2)+size(S_all,2)));
	for c = 1:length(unique(label_all))
		e = sparse(zeros(size(X_all,2)+size(S_all,2),1));
		e(find(label_all==c)) = N{c};
		e(size(X_all,2)+c) = -1;
		M = M + e*e';
	end

	%Locality Matrix
	L = sparse(zeros(size(X_all,2)));
	for c = 1:length(unique(label_all))
		e = sparse(zeros(size(X_all,2),1));
		index = find(label_all==c);
		e(index) = N{c};
		L = L + e*e';
	end	
	L = 2*(diag(sum(L,2)) - L);

	%Build blk Matrix
	T = blkdiag(X_all,S_all);
	L = blkdiag(L,zeros(size(S_all,2)));
	T_all = blkdiag([X test],S_all);

	%Compute Eigen
	G = eye(size(X_all,1)+size(S_all,1));
	[A,~] = eigs(T*(M+lambda2*L)*T'+lambda1*G,T_all*H*T_all',size(S_all,1),'SM');
	As = A(1:size(X,1),:);
	At = A(size(X,1)+1:end,:);
	new_X_test = As'*test;
	new_Y_test = At'*test_Y;
	new_X_test = new_X_test*diag(1./sqrt(sum(new_X_test.^2)));
	new_Y_test = new_Y_test*diag(1./sqrt(sum(new_Y_test.^2)));
	[~,label] = max(new_Y_test'*new_X_test);
	acc = sum(label'==test_label)/length(test_label);
	label = label';

