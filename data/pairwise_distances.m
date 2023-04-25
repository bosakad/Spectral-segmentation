function D = pairwise_distances(X,Y)
%D = PAIRWISE_DISTANCES(X,Y)   Distances between points from two sets
% X and Y, an n-times-d and an m-times-d matrix with points in each row 
% D, n-times-m matriw with pointwise distances

n = size(X,1);
m = size(Y,1);
D = zeros(n,m);
for i = 1:size(X,2)
    D = D + (X(:,i)*ones(1,m)-ones(n,1)*Y(:,i)').^2;    
end