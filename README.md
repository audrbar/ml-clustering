# Clustering
Clustering is an unsupervised machine-learning technique. It is the process of division of the 
dataset into groups in which the members in the same group possess similarities in features. \
k-means clustering is a method of vector quantization, that aims to partition n observations into 
k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers 
or cluster centroid), serving as a prototype of the cluster. This results in a partitioning 
of the data space into Voronoi cells. k-means clustering minimizes within-cluster variances 
(squared Euclidean distances), but not regular Euclidean distances.
The commonly used clustering techniques are:
- K-Means clustering, 
- Hierarchical clustering, 
- Density-based clustering, 
- Model-based clustering. 
We can implement the K-Means clustering machine learning algorithm in the elbow method using 
the scikit-learn library in Python.
## Main aspects
- Elbow method to determine the optimal number of clusters in K-Means using Python;
- Assess the quality of K-Means clusters using Within-Cluster Sum of Squares (WCSS).
## Dimensionality Reduction with Principal Component Analysis (PCA)
PCA is a technique used to emphasize variation and capture strong patterns in a dataset. It transforms the data 
into a new set of variables, the principal components, which are orthogonal (uncorrelated), ensuring that 
the first principal component captures the most variance, and each succeeding one, less so. This transformation 
is not just a mere reduction of dimensions; it’s an insightful distillation of data.
**In the following sections, we will cover:**
- Pre-processing: Preparing your dataset for analysis.
- Scaling: Why and how to scale your data.
- Optimal PCA Components: Determining the right number of components.
- Applying PCA: Transforming your data.
- KMeans Clustering: Grouping the transformed data.
- Analyzing PCA Loadings: Understanding what your components represent.
- From PCA Space to Original Space: Interpreting the cluster centers.
- Centroids and Means: Comparing cluster centers with the original data mean.
- Deep Dive into Loadings: A closer look at the features influencing each principal component.
## Hierarchical clustering
Hierarchical clustering, also known as hierarchical cluster analysis, is an algorithm that groups similar objects into 
groups called clusters.
## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Creates cluster for noise. Density is key.
## Silhouette Score
The overall silhouette score is the average silhouette score for all points in the dataset. It provides a single 
measure of the overall clustering quality.
## Davies-Bouldin score
The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity 
is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less 
dispersed will result in a better score.
## Dunn index
The Dunn index (DI) (introduced by J. C. Dunn in 1974), a metric for evaluating clustering algorithms, is an internal 
evaluation scheme, where the result is based on the clustered data itself. Like all other such indices, the aim of this 
Dunn index to identify sets of clusters that are compact, with a small variance between members of the cluster, and 
well separated, where the means of different clusters are sufficiently far apart, as compared to the within cluster 
variance. 
Higher the Dunn index value, better is the clustering. The number of clusters that maximizes Dunn index is taken as the 
optimal number of clusters k. It also has some drawbacks. As the number of clusters and dimensionality of the data 
increase, the computational cost also increases. 
## Resources
![The Ultimate Step-by-Step Guide to Data Mining with PCA and KMeans](https://drlee.io/the-ultimate-step-by-step-guide-to-data-mining-with-pca-and-kmeans-83a2bcfdba7d)
