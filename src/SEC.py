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



def Stochastic_Ensemble_Consensus(image, labels, r, k, sigma, num_iteration, expectation):
    """
    Post processing for spectral segmentation using Stochastic ensemble consensus
    see https://uwaterloo.ca/vision-image-processing-lab/sites/ca.vision-image-processing-lab/files/uploads/files/enabling_scalable_spectral_clustering_for_image_segmentation_1.pdf
    
    Args:
        image: (N, D, 3) numpy array, N is the number of data points, D is the dimension of data points
        labels: (N, D) numpy array, labels[i, j] is the cluster label of data point i, j
        r: int, parameter for distance matrix
        k: int, number of classes
        sigma: float, parameter for influence matrix
        num_iteration: int, number of iterations
        expectation: bool, if True, use expectation, else use probability
        
    Returns:
        labels: (N, D) numpy array, labels[i, j] is the label of data point i, j
    """
    

    out_label = Graph_Laplacian.get_label_mat(labels, None, r)
    probability, kernel = Graph_Laplacian.get_distance_mat(image, out_label, r)
    influence_mat  = Graph_Laplacian.get_influence_mat(image, kernel, sigma, r)
    prod = None

    # print(image.shape)
    # print(r)
    # print(influence_mat.shape)
    # print(probability.shape)
    # exit()
    # exit()

    if expectation:
        prod = influence_mat * probability

    for _ in range(num_iteration):
        L = []
        if not expectation:
            random_choice = np.random.uniform(0, 1, size=probability.shape)

            prod = (random_choice < probability).astype(np.float32) * influence_mat

            # mask = np.sum(prod, axis=0) > 0

        for j in range(k):
            class_index = (out_label == j)
            
            L.append(np.sum(prod, axis=0, where=class_index))

        
        L = np.stack(L, axis=0)
        if expectation:
            labels = np.argmax(L, axis=0)
        else:
            # labels[mask] = np.argmax(L, axis=0)[mask]
            labels[mask] = np.argmax(np.random.random(L.shape) * np.equal(L,  L.max(axis=0, keepdims=True)), axis=0)
        
        out_label = Graph_Laplacian.get_label_mat(labels, kernel, r)
    
    return labels
