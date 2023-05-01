import skimage
import Spectral_Clustering
import numpy as np
import matplotlib.pyplot as plt
import utility_tools.Preprocessor as Preprocessor
import SEC

def main_clustering():

    pass

    # cluster 1
    # points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=1)
    # labels = spectral_Clustering(points, k=3, sigma=0.25, graphType='unNormalized')


    # labels = spectral_Clustering(points, k=3, sigma=0.1, graphType='randomWalk')


    # cluster 0
    # points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=0)
    # # labels = spectral_Clustering(points, k=3, sigma=0.25, graphType='unNormalized')
    # labels = spectral_Clustering(points, k=3, sigma=0.35, graphType='symmetric')
    # labels = spectral_Clustering(points, k=3, sigma=0.25, graphType='randomWalk')

    # points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=2)
    # # labels = spectral_Clustering(points, k=2, sigma=0.4, graphType='unNormalized')
    # labels = spectral_Clustering(points, k=2, sigma=0.03, graphType='symmetric')
    # labels = spectral_Clustering(points, k=2, sigma=1, graphType='randomWalk')


    # points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=3)    
    # labels = spectral_Clustering(points, k=3, sigma=1, graphType='unNormalized')
    # labels = spectral_Clustering(points, k=3, sigma=0.35, graphType='symmetric')
    # labels = spectral_Clustering(points, k=3, sigma=0.59, graphType='randomWalk')


    # points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=4)    
    # labels = spectral_Clustering(points, k=2, sigma=0.2, graphType='unNormalized')
    # labels = spectral_Clustering(points, k=2, sigma=0.05, graphType='symmetric')
    # labels = spectral_Clustering(points, k=2, sigma=0.12, graphType='randomWalk')


    # Data_loader.plotData(points, labels=labels)


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

    # segment the downscaled image
    labels = Spectral_Clustering.spectral_Segmentation(imgDownScaled, k=2, sigma_i=0.03, sigma_x=6, r=9, graphType='symmetric') # good for plane

    # rescale to the original size
    labels = Preprocessor.rescaleCV2(labels.astype(np.float32), (img.shape[0], img.shape[1]))

    print("Downscaled image segmented!")

    # post processing - improve labels
    new_labels = SEC.Stochastic_Ensemble_Consensus(img, labels.copy(), r=5, k=2, sigma=1, num_iteration=4, expectation=True)


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

    SEC_Segmentation()
    # main_segmentation()
    # main_clustering()