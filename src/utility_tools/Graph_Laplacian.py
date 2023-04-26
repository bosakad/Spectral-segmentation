
import numpy as np
from utility_tools import *
import scipy
import skimage
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

def get_fully_connected_graph_image(image, sigma_i, sigma_x, r):
    """ Compute W matrix of a fully connected graph using Gaussian similarity with parameter sigma
    
    Args:
        image: (N, M) numpy array, N is the number of rows, M is the number of columns
        sigma_i: scalar, standard deviation of Gaussian similarity in intensity domain
        sigma_x: scalar, standard deviation of Gaussian similarity in spatial domain
        r: scalar, radius of spatial domain

    Returns:
        W: (N*M, N*M) numpy array, W[i, j] is the similarity between point i and point j
    
    """

 
 
    image = image[125:145, :20]
    n = image.shape[0]
    m = image.shape[1]
    x = np.arange(0, n)
    y = np.arange(0, m)
    xv, yv = np.meshgrid(x, y)
    points = np.stack([yv, xv], axis=-1).reshape(-1, 2)
    X_distance = np.sum( np.square(points[:, None, :] - points[None, :, :]), axis=-1 )
    intensity = image[points[:,0], points[:,1]]
    I_distance = np.square(intensity.reshape(-1, 1) -  intensity.reshape(1, -1))
    W = np.zeros_like(X_distance)
    W = np.exp(-I_distance / np.square(sigma_i)) * np.exp(-X_distance / np.square(sigma_x)) * (X_distance < np.square(r))
    #print(W.shape)
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

    # points = load_data('../data/spectral_data/points_data.mat', clusterInd=1)

    img = skimage.io.imread('../data/spectral_data/bag.png').astype(np.float32)
    get_fully_connected_graph_image(img, sigma_i=1, sigma_x=1, r=1)
    # W = get_Fully_Connected_Graph(points, sigma=1)

    # L = get_Graph_Laplacian(W, type='unNormalized')
    # L = get_Graph_Laplacian(W, type='symmetric')
    # L = get_Graph_Laplacian(W, type='randomWalk')


    # import Eigen_Solver
    # Eigen_Solver.compute_Eigen_Vectors(L, k=2)


