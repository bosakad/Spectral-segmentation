A = zeros(12,12);
A([1,2,3],[1,2,3]) = 20;
A([4,5,6],[4,5,6]) = 20;
A([7,8,9],[7,8,9]) = 10;
A([10,11,12],[10,11,12]) = 10;
A([1,4],[1,4]) = 8;
A([1,7],[1,7]) = 7;
A([4,10],[4,10]) = 5;
A([7,10],[7,10]) = 5;
A = A .*(1-eye(12));

figure, imagesc(A), axis image

p1 = [false(1,9),true,true,true];
p2 = [false(1,6),true(1,6)];
p3 = [true,true,true,false,false,false,true,true,true,false,false,false];

cut = @(A,p) sum(sum(A(p,~p)));
rcut = @(A,p) cut(A,p)*(1/sum(p)+1/sum(~p));
ncut = @(A,p) cut(A,p)*(1/sum(sum(A(p,:)))+1/sum(sum(A(~p,:))));
mmcut = @(A,p) cut(A,p)*(2/sum(sum(A(p,p)))+2/sum(sum(A(~p,~p))));

[cut(A,p1),cut(A,p2),cut(A,p3)]
[rcut(A,p1),rcut(A,p2),rcut(A,p3)]
[ncut(A,p1),ncut(A,p2),ncut(A,p3)]
[mmcut(A,p1),mmcut(A,p2),mmcut(A,p3)]

%% an additional test
p = randn(1,12)>0;

cAA = sum(sum(A(p,p)))/2;
cAB = sum(sum(A(p,~p)));
cBB = sum(sum(A(~p,~p)))/2;

[ncut(A,p),cAB*(1/(2*cAA+cAB)+1/(2*cBB+cAB))]
[mmcut(A,p),cAB*(1/cAA+1/cBB)]



