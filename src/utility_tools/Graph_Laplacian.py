
import numpy as np
from utility_tools import *
import scipy
from sklearn.neighbors import NearestNeighbors


def get_Fully_Connected_Graph(points, sigma=-1):
    """ Compute W matrix of a fully connected graph using Gaussian similarity with parameter sigma

    Args: 
        points: (N, D) numpy array, N is the number of points, D is the dimension of each point
        sigma: scalar, standard deviation of Gaussian similarity

    Returns:
        W: (N, N) numpy array, W[i, j] is the similarity between point i and point j
    """

    # compute pair-wise squared distances
    W = np.sum( np.square(points[:, None, :] - points[None, :, :]), axis=-1 )


    # if sigma not specified - compute sigma from mean / median of KNN
    if sigma == -1:
        
        n = points.shape[0]
        k = int(np.log(n)) + 1

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
        distances, _ = nbrs.kneighbors(points)

        distances = distances[:, 1:]

        # sigma = distances.mean() * 2
        sigma = scipy.ndimage.median( distances.flatten() ) * 2



    print(sigma)

    # compute similarity
    W = np.exp( - W / (2*np.square(sigma)) )        

    return W


def get_Graph_Laplacian(W, type='unNormalized'):
    """ Compute Graph Laplacian matrix L from similarity matrix W

    Args:
        W: (N, N) numpy array, W[i, j] is the similarity between point i and point j
        type: string, type of Graph Laplacian matrix, can be 'unNormalized', 'symmetric', 'randomWalk'

    Returns:
        L: (N, N) numpy array, L is Graph Laplacian matrix of type 'type'
    """

    # compute degree matrix
    D = np.diag( np.sum(W, axis=1) )

    # compute Graph Laplacian matrix
    if type == 'unNormalized':
    
        L = D - W
    
    elif type == 'symmetric':
        
        I = np.eye(W.shape[0])
        D_sqrt_inv = np.linalg.inv( np.sqrt(D) )
        L = I - D_sqrt_inv @ W @ D_sqrt_inv

    elif type == 'randomWalk':

        I = np.eye(W.shape[0])
        D_inv = np.linalg.inv(D)
        L = I - D_inv @ W

    return L


if __name__ == '__main__':

    points = load_data('../data/spectral_data/points_data.mat', clusterInd=1)

    W = get_Fully_Connected_Graph(points, sigma=1)

    # L = get_Graph_Laplacian(W, type='unNormalized')
    # L = get_Graph_Laplacian(W, type='symmetric')
    L = get_Graph_Laplacian(W, type='randomWalk')


    import Eigen_Solver
    Eigen_Solver.compute_Eigen_Vectors(L, k=2)


