clear
close all
addpath('spectral_functions')

% I = imread('plane.jpg');
% I = double(imresize(I,0.2)); 
% k = 2;

% I = imread('onion.png');
% I = double(imresize(I,0.4)); 
% k = 5;
 
I = imread('peppers.png');
I = double(imresize(I,0.2)); 
k = 5;


[X,Y] = ndgrid(1:size(I,1),1:size(I,2));
P = pairwise_distances([X(:),Y(:)],[X(:),Y(:)]);
S = pairwise_distances(reshape(I,[],3),reshape(I,[],3));

sigmaS = 500;
sigmaP = 500;
r = inf;

W = exp(-S/sigmaS).*exp(-P/sigmaP).*(P<r);

vec = normalized_cut_from_W(W,k);
seg = kmeans_discretize(vec);

SEG = reshape(seg,[size(I,1),size(I,2)]);
K = reshape(kmeans(reshape(I,[],3),k),[size(I,1),size(I,2)]); % k-means

figure
subplot(231), imagesc(P), axis image, title('spatial distances')
subplot(232), imagesc(S), axis image, title('intensity distances')
subplot(234), imagesc(exp(-P/sigmaP),[0,1]), axis image, title('proximity')
subplot(235), imagesc(exp(-S/sigmaS),[0,1]), axis image, title('similarity')
subplot(236), imagesc(W,[0,1]), axis image, title('affinity')

%%
figure
subplot(221), imagesc(uint8(I)), axis image, title('input')
vec2 = reshape(vec(:,2),[size(I,1),size(I,2)]);
vec2 = repmat(uint8(255*(vec2-min(vec2(:)))/(max(vec2(:))-min(vec2(:)))),[1 1 3]);
subplot(222), imagesc(vec2), axis image, title('second eigenvector')
subplot(223), imagesc(repmat(uint8(255*(SEG-1)/(k-1)),[1 1 3])), axis image, title('segmentation'), colormap gray
subplot(224), imagesc(repmat(uint8(255*(K-1)/(k-1)),[1 1 3])), axis image, title('k-means'), colormap gray








