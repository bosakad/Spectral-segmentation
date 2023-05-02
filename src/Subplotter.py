import skimage
import Spectral_Clustering
import numpy as np
import matplotlib.pyplot as plt
import PostProcessor
from utility_tools import Data_loader, Preprocessor

def clustering():

    points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=4) 

    labels1 = Spectral_Clustering.spectral_Clustering(points, k=2, graphType='symmetric')


    # cluster 2
    points2 = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=2)

    labels2 = Spectral_Clustering.spectral_Clustering(points2, k=2, graphType='symmetric')
    
    numClusters = 2
    plt.rcParams['figure.figsize'] = [15, 7]
    subplot, ax = plt.subplots(1, 2)


    # scatter each cluster
    for i in range(numClusters):
        clusterInd = np.where(labels1 == i)
        ax[0].scatter(points[clusterInd, 0], points[clusterInd, 1])

    # scatter each cluster
    for i in range(numClusters):
        clusterInd = np.where(labels2 == i)
        ax[1].scatter(points2[clusterInd, 0], points2[clusterInd, 1])

    plt.savefig("clustersNoTitle.png", dpi=900)
    plt.show()


def main_segmentation():

    """
    Image segmenetation using spectral clustering

    """

    img = skimage.io.imread('../data/spectral_data/plane.jpg').astype(np.float32)
    # img = skimage.io.imread('../data/spectral_data/hair.png').astype(np.float32)
    # img = skimage.io.imread('../data/spectral_data/test_blob_uniform.png').astype(np.float32)
    # img = skimage.io.imread('../data/spectral_data/onion.png').astype(np.float32)
    # img = skimage.io.imread('../data/spectral_data/peppers.png').astype(np.float32)
    img = img / 255

    print(img.shape)

    # img = img[500:900, 500:900, :]

    # plt.imshow(img)
    # plt.show()

    # rescale
    # img = Preprocessor.maxPoolImage_3d(img, k=6)
    # img = Preprocessor.MeanImage_3d(img, k=6)

    k = 1/6
    imgDownScaled = Preprocessor.rescale(img, k)


    # print(img.shape)

    # plt.imshow(imgDownScaled)
    # plt.show()

    # labels = Spectral_Clustering.spectral_Segmentation(img, k=2, sigma_i=0.02, sigma_x=1.5, r=2, graphType='symmetric')
    labels = Spectral_Clustering.spectral_Segmentation(imgDownScaled, k=2, sigma_i=0.03, sigma_x=6, r=9, graphType='symmetric') # good for plane
    # labels = Spectral_Clustering.spectral_Segmentation(imgDownScaled, k=2, sigma_i=0.0005, sigma_x=30, r=40, graphType='symmetric') # hair


    # upscale - how to do that?
    labels = Preprocessor.rescale(labels, 1/k)


    print(labels.shape)
    print(img.shape)

    # plot
    subplot, ax = plt.subplots(1, 3)

    # original image
    ax[0].imshow(img, cmap='gray')   
    # segmentation 
    ax[1].imshow(labels, cmap='gray')
    
    # segmented image
    imgSegmented = img.copy()
    pos = np.where(labels == 0)
    
    col1 = pos[0][:] < img.shape[0]
    col2 = pos[1][:] < img.shape[1]

    range = np.where((col1 & col2) == True)
    xx = pos[0][range]
    yy = pos[1][range]

    imgSegmented[xx, yy, :] = 0
    ax[2].imshow(imgSegmented, cmap='gray') 
    plt.show()
    





def SEC_Segmentation():

    """
    Image segmenetation using spectral clustering with added Stochastic_Ensemble_Consensus

    """

    # load image
    img = skimage.io.imread('../data/spectral_data/plane.jpg').astype(np.float32)
    img = img / 255

    # scaling factor
    k = 1/8

    # downscale
    imgDownScaled = Preprocessor.rescaleSkimage(img, k)

    print("Segmenting Downscaled Image!")

    # segment the downscaled image
    labels = Spectral_Clustering.spectral_Segmentation(imgDownScaled, k=2, sigma_i=0.03, sigma_x=6, r=9, graphType='symmetric') # good for plane

    # rescale to the original size
    labels = Preprocessor.rescaleCV2(labels.astype(np.float32), (img.shape[0], img.shape[1]))

    print("Postprocessing using Stochastic Ensemble Consensus!")

    # post processing - improve labels
    new_labels = PostProcessor.Stochastic_Ensemble_Consensus(img, labels.copy(), r=14, k=2, sigma=0.06, num_iteration=10, expectation=True)


    ################ plot ################### 

    subplot, ax = plt.subplots(1, 3)

    # original image
    ax[0].imshow(img)   

    # before preprocessing
    imgSegmented = img.copy()
    boundry = skimage.segmentation.find_boundaries(labels, mode='thick')
    imgSegmented[boundry, :] = [1, 0, 0]

    ax[1].imshow(imgSegmented) 
    
    # after preprocessing
    imgSegmented2 = img.copy()
    boundry = skimage.segmentation.find_boundaries(new_labels, mode='thick')
    imgSegmented2[boundry, :] = [1, 0, 0]

    ax[2].imshow(imgSegmented2) 
    
    plt.show()




if __name__ == '__main__':

    # SEC_Segmentation()
    # main_segmentation()
    main_clustering()