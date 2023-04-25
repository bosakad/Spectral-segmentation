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



def plotData(points):
    """ Plot data
    """

    plt.scatter(points[:, 0], points[:, 1])
    plt.show()




if __name__ == '__main__':

    # Load data
    # points = load_data('./data/spectral_data/points_data.mat', clusterInd=0)

    points = load_data('./data/spectral_data/points_data.mat', clusterInd=3)

    print(points.shape)

    # Plot data
    plotData(points)

