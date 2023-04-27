import scipy
import numpy as np


def compute_Eigen_Vectors(L, k):
    """ Compute first k eigen vectors of Graph Laplacian matrix L

    Args:
        L: (N, N) numpy array, L is Graph Laplacian matrix
        k: int, number of eigen vectors to compute = number of clusters
    """

    # compute eigen vectors

    # use the fact that the matrix is symmetric positive definite
    # IMPORTANT: if needs to use L matrix after - remove the overwrite_a=True !!
    eigen_values, eigen_vectors = scipy.linalg.eigh(L, subset_by_index=[0, k-1], check_finite=False, overwrite_a=True) 

    return eigen_values, eigen_vectors

def compute_Eigen_Sparse_Vectors(L, k):
    """ Compute first k eigen vectors of Graph Laplacian matrix L

    Args:
        L: (N, N) numpy array, L is Graph Laplacian matrix
        k: int, number of eigen vectors to compute = number of clusters
    """

    # compute eigen vectors

    # use the fact that the matrix is symmetric positive definite
    # IMPORTANT: if needs to use L matrix after - remove the overwrite_a=True !!
    # change tol value for more speed vs accuracy
    eigen_values, eigen_vectors = scipy.sparse.linalg.eigsh(L, k=k, which='SM', tol=1e-1) 

    return eigen_values, eigen_vectors

    
