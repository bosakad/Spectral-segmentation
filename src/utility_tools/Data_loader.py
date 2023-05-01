import numpy as np
import scipy
import matplotlib.pyplot as plt


def load_data(fileName, clusterInd=0):
    """ Load data from .mat files
    """

    if fileName.endswith('.mat'):
        mat = scipy.io.loadmat(fileName)
        points = np.array(mat['points'])[0][clusterInd]

    return points



def plotData(points, labels=np.array([]), title=None):
    """ Plot data
    """

    if labels.size == 0:
        plt.scatter(points[:, 0], points[:, 1])
        plt.show()

    else:

        numClusters = np.max(labels) + 1

        # scatter each cluster
        for i in range(numClusters):
            clusterInd = np.where(labels == i)
            plt.scatter(points[clusterInd, 0], points[clusterInd, 1])

        if title is not None:
            plt.title(title) 
    
        plt.show()





if __name__ == '__main__':

    # Load data
    # points = load_data('./data/spectral_data/points_data.mat', clusterInd=0)

    points = load_data('../data/spectral_data/points_data.mat', clusterInd=3)

    print(points.shape)

    # Plot data
    plotData(points)

