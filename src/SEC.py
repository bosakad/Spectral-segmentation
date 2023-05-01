import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import Spectral_Clustering
from utility_tools import Data_loader, Graph_Laplacian, Eigen_Solver

def Stochastic_Ensemble_Consensus(original_Image, labels):
    """
    Stochastic_Ensemble_Consensus to obtain the final segmentation

    Args:
        original_Image: N*M image
        labels: N*M labels
    
    Returns:
        labels: N*M labels after SEC

    """

    # specify parameters
    windowSize = 5
    sigma = 0.05
    numberOfIterations = 1

    # get the space probability density function and the influence matrix
    influenceM, DistanceMatrix = Graph_Laplacian.get_DistanceM_InfluenceMatrix(original_Image, windowSize, sigma)


    indices = DistanceMatrix.nonzero()[0]
    probM = 1 / DistanceMatrix[indices].reshape((-1, windowSize**2))

    print(probM.shape)

    # make the segmentation better
    for i in range(numberOfIterations):

        # draw from probM

        pass
        


        # print(indices[0][0])
        # print(probM[0])
        # print(indices[1].shape)





def experiments():
    img = skimage.io.imread('../data/spectral_data/plane.jpg').astype(np.float32)
    img = img / 255

    if len(img.shape) == 3:
        rgb = True
        img = img[0:300, 0:300, :]


    labels = Spectral_Clustering.spectral_Segmentation(img, k=2, sigma_i=0.03, sigma_x=2, r=3, graphType='symmetric')

    subplot, ax = plt.subplots(1, 3)

    ax[0].imshow(img, cmap='gray')    
    ax[1].imshow(labels, cmap='gray')

    imgSegmented = img.copy()
    imgSegmented[labels == 0] = 0
    ax[2].imshow(imgSegmented, cmap='gray') 

    plt.show()
