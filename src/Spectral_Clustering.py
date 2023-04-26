import numpy as np
import sklearn.cluster

from utility_tools import Data_loader, Graph_Laplacian, Eigen_Solver

def spectral_Clustering(data, k, sigma, graphType='unNormalized'):
    """ Spectral Clustering algorithm
    
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
    # W = Graph_Laplacian.get_Fully_Connected_Graph(data)
    L = Graph_Laplacian.get_Graph_Laplacian(W, type=graphType)

    # compute k smallest eigen vectors
    eigen_values, eigen_vectors = Eigen_Solver.compute_Eigen_Vectors(L, k=k)

    # cluster the data
    kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init=10).fit(eigen_vectors)

    labels = kmeans.labels_

    return labels







if __name__ == "__main__":

    # cluster 1
    # points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=1)
    # labels = spectral_Clustering(points, k=3, sigma=0.25, graphType='unNormalized')
    # labels = spectral_Clustering(points, k=3, sigma=0.35, graphType='symmetric')
    # labels = spectral_Clustering(points, k=3, sigma=0.59, graphType='randomWalk')


    # cluster 0
    points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=0)
    # # labels = spectral_Clustering(points, k=3, sigma=0.25, graphType='unNormalized')
    labels = spectral_Clustering(points, k=3, sigma=0.35, graphType='symmetric')
    # labels = spectral_Clustering(points, k=3, sigma=0.25, graphType='randomWalk')

    # points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=2)
    # doesnt work ???
    # labels = spectral_Clustering(points, k=2, sigma=0.4, graphType='unNormalized')
    # labels = spectral_Clustering(points, k=2, sigma=0.03, graphType='symmetric')
    # labels = spectral_Clustering(points, k=2, sigma=1, graphType='randomWalk')


    # points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=3)    
    # labels = spectral_Clustering(points, k=3, sigma=1, graphType='unNormalized')
    # labels = spectral_Clustering(points, k=3, sigma=0.35, graphType='symmetric')
    # labels = spectral_Clustering(points, k=3, sigma=0.59, graphType='randomWalk')


    # points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=4)    
    # labels = spectral_Clustering(points, k=2, sigma=0.2, graphType='unNormalized')
    # labels = spectral_Clustering(points, k=2, sigma=0.1, graphType='symmetric')
    # labels = spectral_Clustering(points, k=2, sigma=0.12, graphType='randomWalk')


    Data_loader.plotData(points, labels=labels)





