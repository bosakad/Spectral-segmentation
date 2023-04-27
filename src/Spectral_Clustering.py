import numpy as np
import sklearn.cluster
import skimage
import matplotlib.pyplot as plt

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
    # W =  Graph_Laplacian.get_fully_connected_graph_image(data, sigma_i=1, sigma_x=1, r=5)
    # W = Graph_Laplacian.get_Fully_Connected_Graph(data)   
    L = Graph_Laplacian.get_Graph_Laplacian(W, type=graphType)

    # compute k smallest eigen vectors
    eigen_values, eigen_vectors = Eigen_Solver.compute_Eigen_Vectors(L, k=k)


    print(eigen_vectors[:, 1][:10])


    # cluster the data
    kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init=10).fit(eigen_vectors)

    labels = kmeans.labels_

    # TODO: normalize over rows!!! in the L_sym

    # print(eigen_vectors[:, 1] [np.where(labels == 0)] )

    # print(eigen_vectors[:, 1] [np.where(labels == 1)] )

    # print(eigen_vectors[:, 2] [np.where(labels == 2)] )


    return labels


def spectral_Segmentation(image, k, sigma_i, sigma_x, r,  graphType='unNormalized'):
    """ Spectral Clustering algorithm

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


    # create graph of similarites
    # W = Graph_Laplacian.get_Fully_Connected_Graph(data, sigma=sigma)
    W =  Graph_Laplacian.get_connected_graph_image(image, sigma_i=1, sigma_x=1, r=5)
    # W = Graph_Laplacian.get_Fully_Connected_Graph(data)   
    L = Graph_Laplacian.get_Sparse_Graph_Laplacian(W, type=graphType)

    # compute k smallest eigen vectors
    eigen_values, eigen_vectors = Eigen_Solver.compute_Eigen_Sparse_Vectors(L, k=k)


    print(eigen_vectors[:, 1][:10])


    # cluster the data
    # kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init=10).fit(eigen_vectors)

    # labels = kmeans.labels_

    # TODO: normalize over rows!!! in the L_sym

    # print(eigen_vectors[:, 1] [np.where(labels == 0)] )

    # print(eigen_vectors[:, 1] [np.where(labels == 1)] )

    # print(eigen_vectors[:, 2] [np.where(labels == 2)] )


    return None






if __name__ == "__main__":

    img = skimage.io.imread('../data/spectral_data/bag.png').astype(np.float32)

    # cluster 1
    # points = Data_loader.load_data('../data/spectral_data/points_data.mat', clusterInd=1)
    # labels = spectral_Clustering(points, k=3, sigma=0.25, graphType='unNormalized')
    print(img.max())
    img = img / 255
    labels = spectral_Segmentation(img, k=3, sigma_i=0.4, sigma_x=3, r=5, graphType='symmetric')
    plt.imshow(labels.reshape(20, 20))
    plt.show()
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





