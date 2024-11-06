from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
data = iris.data

# Step 1: Plot Dendrogram
plt.figure(figsize=(10, 7))
linked = linkage(data, method='ward')
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering on Iris Dataset")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Step 2: Calculate WCSS for different numbers of clusters
wcss = []
max_clusters = 10
for k in range(1, max_clusters + 1):
    agglomerative = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agglomerative.fit_predict(data)

    # Calculate WCSS for the clusters formed by Agglomerative Clustering
    centroids = []
    for cluster_id in np.unique(labels):
        cluster_points = data[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Calculate WCSS
    _, dists = pairwise_distances_argmin_min(data, centroids)
    wcss.append(np.sum(dists ** 2))

# Step 3: Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal K in Agglomerative Clustering on Iris Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
