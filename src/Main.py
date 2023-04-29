import skimage
import Spectral_Clustering
import numpy as np
import matplotlib.pyplot as plt
import utility_tools.Preprocessor as Preprocessor

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

    img = skimage.io.imread('../data/spectral_data/plane.jpg').astype(np.float32)
    # img = skimage.io.imread('../data/spectral_data/test_blob_uniform.png').astype(np.float32)
    # img = skimage.io.imread('../data/spectral_data/onion.png').astype(np.float32)
    # img = skimage.io.imread('../data/spectral_data/peppers.png').astype(np.float32)
    img = img / 255

    # img = img[0:140, 0:140]

    # img = skimage.color.gray2rgb(img)
    # print(img.shape)


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
    labels = Spectral_Clustering.spectral_Segmentation(imgDownScaled, k=4, sigma_i=0.04, sigma_x=2, r=2, graphType='symmetric')


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
    pos = np.where(labels != 0)
    
    col1 = pos[0][:] < img.shape[0]
    col2 = pos[1][:] < img.shape[1]

    range = np.where((col1 & col2) == True)
    xx = pos[0][range]
    yy = pos[1][range]

    imgSegmented[xx, yy, :] = 0
    ax[2].imshow(imgSegmented, cmap='gray') 
    plt.show()
    










if __name__ == '__main__':

    main_segmentation()
    main_clustering()