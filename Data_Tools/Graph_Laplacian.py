import numpy as np
import Data_loader

def get_Fully_Connected_Graph(points, sigma=1):
    """ Compute W matrix of a fully connected graph using Gaussian similarity with parameter sigma

    Args: 
        points: (N, D) numpy array, N is the number of points, D is the dimension of each point
        sigma: scalar, standard deviation of Gaussian similarity

    Returns:
        W: (N, N) numpy array, W[i, j] is the similarity between point i and point j
    """

    # compute pair-wise squared distances
    W = np.sum( np.square(points[:, None, :] - points[None, :, :]), axis=-1 )

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

    points = Data_loader.load_data('./data/spectral_data/points_data.mat', clusterInd=3)

    W = get_Fully_Connected_Graph(points, sigma=1)

    # L = get_Graph_Laplacian(W, type='unNormalized')
    # L = get_Graph_Laplacian(W, type='symmetric')
    L = get_Graph_Laplacian(W, type='randomWalk')


    


