clear
close all
addpath('spectral_functions')

I = imread('bag.png');
I = double(I(221:250,1:30)); 

[X,Y] = ndgrid(1:size(I,1),1:size(I,2));
P = pairwise_distances([X(:),Y(:)],[X(:),Y(:)]);
S = pairwise_distances(I(:),I(:));

sigmaS = 500;
sigmaP = 100;
r = inf;
k = 2;

W = exp(-S/sigmaS).*exp(-P/sigmaP).*(P<r);

vec = normalized_cut_from_W(W,k);
seg = kmeans_discretize(vec);

SEG = reshape(seg,size(I));

figure
subplot(231), imagesc(P), axis image, title('spatial distances')
subplot(232), imagesc(S), axis image, title('intensity distances')
subplot(234), imagesc(exp(-P/sigmaP),[0,1]), axis image, title('proximity')
subplot(235), imagesc(exp(-S/sigmaS),[0,1]), axis image, title('similarity')
subplot(236), imagesc(W,[0,1]), axis image, title('affinity')

figure
subplot(131), imagesc(I), colormap gray, axis image, title('input')
subplot(132), imagesc(reshape(vec(:,2),size(I))), axis image, title('second eigenvector')
subplot(133), imagesc(SEG), axis image, title('segmentation')








