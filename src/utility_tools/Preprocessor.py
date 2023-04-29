import skimage
import numpy as np

def maxPoolImage_3d(image, k):
    """ Max pools image by a factor of k to make it smaller then segment

    Args:
        image: (N, M, 3) numpy array, image to max pool
        k: int, factor to max pool image by

    Returns:
        image: (N, M) numpy array, max pooled image
    """

    # max pool image
    for ch in range(image.shape[2]):
        reducedImageChannel = skimage.measure.block_reduce(image[:, :, ch], (k, k), np.max)

        if ch == 0:
            reducedImage = np.zeros((reducedImageChannel.shape[0], reducedImageChannel.shape[1], 3))
        
        reducedImage[:, :, ch] = reducedImageChannel

    return reducedImage



def MeanImage_3d(image, k):
    """ Means image by a factor of k to make it smaller then segment

    Args:
        image: (N, M, 3) numpy array, image to max pool
        k: int, factor to max pool image by

    Returns:
        image: (N, M) numpy array, max pooled image
    """

    # max pool image
    for ch in range(image.shape[2]):
        reducedImageChannel = skimage.measure.block_reduce(image[:, :, ch], (k, k), np.mean)

        if ch == 0:
            reducedImage = np.zeros((reducedImageChannel.shape[0], reducedImageChannel.shape[1], 3))
        
        reducedImage[:, :, ch] = reducedImageChannel

    return reducedImage


def rescale(image, k):
    """ Rescale the image by a factor of k
    
    Args:
        image: (N, M, 3) numpy array, image to max pool
        k: int, factor to max pool image by

    Returns:    
        image_rescaled: (N, M, 3) numpy array, rescaled image
    """

        # check type of the image
    if len(image.shape) == 3:  rgb = True
    else:                      rgb = False

    if rgb:
        image_rescaled = skimage.transform.rescale(image, k, channel_axis=2)
    else:
        image_rescaled = skimage.transform.rescale(image, k)

    return image_rescaled





