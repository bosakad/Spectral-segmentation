close all
clear
addpath('spectral_functions')
load('spectral_data/points_data')
k = [3,3,2,3,2];
sigma = [0.1 0.1 0.001 0.1 0.01];

for d = 1:5   
    
    pointsd = points{d};
    sigmad = sigma(d);
    kd = k(d);
    
    %% normalized cut with gaussian full affinity W
    DW = pairwise_distances(pointsd,pointsd);
    W = exp(-DW/sigmad);
    vec = normalized_cut_from_W(W,kd);
    seg = kmeans_discretize(vec);
    
    %% visualization
    col = 'rgbcmy';
    per = []; % placeholder for sorted permutation
    figure
    subplot(231), plot(pointsd(:,1),pointsd(:,2),'k.'), axis equal, title('input')
    subplot(234), hold on
    for i=1:kd
        ic = find(seg==i);
        plot(pointsd(ic,1),pointsd(ic,2),[col(i),'.'])
        per = [per;ic];
    end
    axis equal, title('clusters')
    
    subplot(232), plot(vec(:,2),'.'), title('second eigenvector')
    subplot(235), plot(vec(per,2),'.'), title('second eigenvector sorted')
    
    subplot(233), imagesc(W), axis square, title('affinity matrix')
    subplot(236), imagesc(W(per,per)), axis square, title('affinity matrix sorted')
    
end