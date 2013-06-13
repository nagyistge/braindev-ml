braindev-ml
===========

Developmental MRI and machine learning combined

#Feature reduction algorithms

We have two proposed pipelines for initial reduction of features. 

##First, appropriate for fMRI o DTI.

1.Superpixel reduction.  This is an algorithm used in image recognition than
lumps pixels together into lumps based on the intensity of the image. This is
an unsupervised algorithm.

2.Compute mass univariate correlaion map at group level with the outcome.

3.Markov Random Field segmentation based on the group correlation map.

4.Apply MRF segmentation to the individual images.

5. Do one of the following two clustering algorithms:

 1. Do a clustering of regions based on the correlation matrix of the clusters
    from the MRF. 

 2. Kernel PCA or similar methods. This allows for clustering non linear
    relationships between the factors.

6. Univariate or RFE feadure selection.

##Secondly probably most suited for fMRI

1. Instead of step one and two. Calculate a similarity metric for each voxel
   (ex MI) compared to the surrounding voxles.  KL-divergence , MI

2. Apply the MRF clustering based on this map.

3. After the MRF clusterng same to previous pipeline.

#Model

1. Gaussian processes
	Try to reduce features enough so that a 2nd or 3rd degree polynomial
function can be fit. This also allows for calculating a confidence interval for
the prediction.

2. SVR
	Compare with the currently used model that is working ok. Also try to
expand with 2, 3rd degree polynomial kernels.

#Modality integration

0. (experimental) For each cluster calculate a connection probability map and make
DTI clusters based on a cutoff of connection probability.

1. Concatenate the vectors after feature reduction

2. secondary model after trainng the initial (fails to capture cross modality
relationships)

3. Combine at an earlier level, ex just after removing the spatial informaton.

#Model human readability

1. Show information containing areas after optimal feature reduction

2. Show model deterioration after removing important areas.

3. Successively simplify the model forcefully until it is human readable.
	ex. force fewer clusters until you have so few clusters so you can plot
easy graphs.


#Dependencies

##Python
numpy
scipy
joblib
sklearn
nibabel

##C++



