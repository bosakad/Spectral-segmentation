
import numpy as np
from utility_tools import *
import scipy
import skimage
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import bsr_matrix
import torch


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

def get_connected_graph_image(image, sigma_i, sigma_x, r):
    """ Compute W matrix of a fully connected graph using Gaussian similarity with parameter sigma
    
    Args:
        image: (N, M) numpy array, N is the number of rows, M is the number of columns
        sigma_i: scalar, standard deviation of Gaussian similarity in intensity domain
        sigma_x: scalar, standard deviation of Gaussian similarity in spatial domain
        r: scalar, radius of spatial domain

    Returns:
        W: (N*M, N*M) numpy array, W[i, j] is the similarity between point i and point j
    
    """

    n = image.shape[0]
    m = image.shape[1]
    x = np.arange(0, m)
    y = np.arange(0, n)
    xv, yv = np.meshgrid(x, y)
    p_xv = np.pad(xv, r, constant_values=-1)
    p_yv = np.pad(yv, r, constant_values=-1)
    index = torch.FloatTensor(xv + m * yv).view(1, n, m)
    p_index = torch.FloatTensor(p_xv + m * p_yv).view(1, n + 2 * r, m + 2 * r)
    p_image = torch.FloatTensor(np.pad(image, r, constant_values=-1)).unsqueeze(0)
    xv = torch.FloatTensor(p_xv).view(1, n + 2 * r, m + 2 * r)
    yv = torch.FloatTensor(p_yv).view(1, n + 2 * r, m + 2 * r)
    size = 2 * r + 1
    numK = size ** 2
    kernel = torch.zeros(numK, 1, size, size, dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            kernel[i * size + j, 0, i, j] = 1
    
    with torch.no_grad():
        out_x = torch.nn.functional.conv2d(xv, kernel)
        out_y = torch.nn.functional.conv2d(yv, kernel)
        out_image = torch.nn.functional.conv2d(p_image, kernel).numpy()
        col = torch.nn.functional.conv2d(p_index, kernel).view(-1).numpy().astype(np.int32)
        row = index.squeeze(0).tile(numK, 1, 1).view(-1).numpy().astype(np.int32)
        intensity_data = (np.square(out_image - np.expand_dims(image, 0))).reshape(-1)
        distance_data = (torch.square(out_x - xv[:, r:-r,r:-r]) + torch.square(out_y - yv[:, r:-r,r:-r])).view(-1).numpy()
    
    data = np.exp(-intensity_data / np.square(sigma_i)) * np.exp(-distance_data / np.square(sigma_x))
    W = bsr_matrix((data[col >= 0], (row[col >= 0], col[col >= 0])), shape=(n * m, n * m), dtype=np.float32)
    
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


def get_Sparse_Graph_Laplacian(W : bsr_matrix, type='unNormalized'):
    """ Compute Graph Laplacian matrix L from similarity matrix W

    Args:
        W: (N, N) numpy array, W[i, j] is the similarity between point i and point j
        type: string, type of Graph Laplacian matrix, can be 'unNormalized', 'symmetric', 'randomWalk'

    Returns:
        L: (N, N) numpy array, L is Graph Laplacian matrix of type 'type'
    """

    assert(W.shape[0] == W.shape[1])
    diag_elements = np.asarray(scipy.sum(W, axis=1)).reshape(-1)
    print(diag_elements.shape)
    L = None

    # compute Graph Laplacian matrix
    if type == 'unNormalized':
        D = scipy.sparse.diags(diag_elements, format='bsr')
        L = D - W
    
    elif type == 'symmetric':
        I = scipy.sparse.eye(W.shape[0], format='bsr')
        sqrt_diag_elements = np.sqrt(diag_elements)
        inv_sqrt_diag_elements = 1 / sqrt_diag_elements
        D_sqrt_inv = scipy.sparse.diags(inv_sqrt_diag_elements, format='bsr')
        L = I - D_sqrt_inv @ W @ D_sqrt_inv

    elif type == 'randomWalk':
        I = scipy.sparse.eye(W.shape[0], format='bsr')
        inv_diag_elements = 1 / diag_elements
        D_inv = scipy.sparse.diags(inv_diag_elements, format='bsr')
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


