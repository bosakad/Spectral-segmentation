import numpy as np
import sklearn.preprocessing
import sklearn.cluster
import skimage
import matplotlib.pyplot as plt

from utility_tools import Data_loader, Graph_Laplacian, Eigen_Solver

def spectral_Clustering(data, k, sigma=-1, graphType='unNormalized'):
    """ Spectral Clustering algorithm for data clustering
    
    Args:
        data: (N, D) numpy array, N is the number of data points, D is the dimension of data points
        k: int, number of clusters
        sigma: float, parameter for similarity matrix
        graphType: string, type of Graph Laplacian matrix, can be 'unNormalized', 'symmetric', 'randomWalk'

    Returns:
        labels: (N, ) numpy array, labels[i] is the cluster label of data point i
    """


    # create graph of similarites
    W = Graph_Laplacian.get_Fully_Connected_Graph(data, sigma=sigma)

    # create graph Laplacian
    L = Graph_Laplacian.get_Graph_Laplacian(W, type=graphType)

    # compute k smallest eigen vectors
    _, eigen_vectors = Eigen_Solver.compute_Eigen_Vectors(L, k=k)

    # normalize over rows
    eigen_vectors = sklearn.preprocessing.normalize(eigen_vectors, norm='l2', axis=1, copy=False)

    # cluster the data
    kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init=10).fit(eigen_vectors)
    labels = kmeans.labels_

    return labels


def spectral_Segmentation(image, k, sigma_i, sigma_x, r,  graphType='unNormalized'):
    """ Spectral Clustering algorithm image segmentation

    Args:   
        image:  
        k: int, number of clusters
        sigma_i: float, parameter for similarity matrix
        sigma_x: float, parameter for similarity matrix
        r: int, parameter for similarity matrix
        graphType: string, type of Graph Laplacian matrix, can be 'unNormalized', 'symmetric', 'randomWalk'
    
    Returns:    
        labels: (N, ) numpy array, labels[i] is the cluster label of data point i


    """

    # check type of the image
    if len(image.shape) == 3:  rgb = True
    else:                      rgb = False


    # create graph of similarites
    if rgb:
        W =  Graph_Laplacian.get_connected_graph_image_3D(image, sigma_i=sigma_i, sigma_x=sigma_x, r=r)
        
    else:
        W =  Graph_Laplacian.get_connected_graph_image(image, sigma_i=sigma_i, sigma_x=sigma_x, r=r)
        
    
    L = Graph_Laplacian.get_Sparse_Graph_Laplacian(W, type=graphType)
    
    L = sklearn.preprocessing.normalize(L, norm='l2', axis=1, copy=False) # normalize over rows

    # compute k smallest eigen vectors
    eigen_values, eigen_vectors = Eigen_Solver.compute_Eigen_Sparse_Vectors(L, k=k)

    # normalize over rows
    eigen_vectors = sklearn.preprocessing.normalize(eigen_vectors, norm='l2', axis=1, copy=False)

    # print(eigen_vectors[:, 1][:10])

    # cluster the data
    kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init=20).fit(eigen_vectors)
    labels = kmeans.labels_


    return labels.reshape((image.shape[0], image.shape[1]))



    














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





