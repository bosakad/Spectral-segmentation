import utility_tools.Graph_Laplacian as Graph_Laplacian


def Stochastic_Ensemble_Consensus(original_Image, labels):
    """
    Stochastic_Ensemble_Consensus to obtain the final segmentation

    Args:
        original_Image: N*M image
        labels: N*M labels
    
    Returns:
        labels: N*M labels after SEC

    """

    # specify parameters
    windowSize = 5
    sigma = 0.05
    numberOfIterations = 1

    # get the space probability density function and the influence matrix
    influenceM, DistanceMatrix = Graph_Laplacian.get_DistanceM_InfluenceMatrix(original_Image, windowSize, sigma)


    indices = DistanceMatrix.nonzero()[0]
    probM = 1 / DistanceMatrix[indices].reshape((-1, windowSize**2))

    print(probM.shape)

    # make the segmentation better
    for i in range(numberOfIterations):

        # draw from probM

        pass
        


        # print(indices[0][0])
        # print(probM[0])
        # print(indices[1].shape)





