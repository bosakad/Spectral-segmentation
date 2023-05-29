# Spectral Clustering and Segmentation

We implemented Spectral Clustering which is a popular unsupervised clustering technique suitable for non-convex dataset. 
We then use this method to segment grayscale and RGB images. To make the clustering more efficient for images, 
we investigate more into parameter tuning and parameter estimation using minimal spanning trees and K-NN distances, 
different types of graph Laplacians, sparse matrix representation, upscaling, downscaling and post-processing steps based on Stochastic ensemble consensus that can improve the performance at low-quality cost.

Spectral Clustering | Image Segmentation
--- | ---
![](https://github.com/bosakad/Spectral-segmentation/assets/87597050/4c946138-890b-4605-83f4-445647404ce4) | ![](https://github.com/bosakad/Spectral-segmentation/assets/87597050/82c200cc-a126-4d53-9d56-697cd2695328)


### Set up: ###
`pip install -r requirements.txt`

### Please see: ###
#### Examples: ####

[Spectral Clustering](https://github.com/bosakad/Spectral-segmentation/blob/main/src/Spectral_Clustering.ipynb)   
[Image Segmentation using Spectral Clustering](https://github.com/bosakad/Spectral-segmentation/blob/main/src/Spectral_Segmentation.ipynb)

#### References: ####

[A Tutorial on Spectral Clustering](https://arxiv.org/abs/0711.0189)  
[Normalized cuts and image segmentation](https://ieeexplore.ieee.org/abstract/document/868688?casa_token=GSpu1dgcN3UAAAAA:NSn5NK6LtNB2FfkVqG9d-EQf8C6iBzXlfIQaDtxcZW5O3mtmgXK6XR5ZmyGjBMkgngo1J_JwAA)  
[Stochastic ensemble consensus](https://uwaterloo.ca/vision-image-processing-lab/publications/sec-stochastic-ensemble-consensus-approach-unsupervised-sar)  
[Enabling scalable spectral clustering for image segmentation](https://www.sciencedirect.com/science/article/abs/pii/S0031320310003110)  

### Authors: ###

[bosakad](https://github.com/bosakad)   
[harshnehal1996](https://github.com/harshnehal1996)
