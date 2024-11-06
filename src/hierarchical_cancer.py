from sklearn.datasets import load_breast_cancer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min, accuracy_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = load_breast_cancer()
X = data.data
y_true = data.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Plot Dendrogram
plt.figure(figsize=(10, 7))
linked = linkage(X_scaled, method='ward')
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering on Breast Cancer Dataset")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Step 2: Calculate WCSS for different numbers of clusters
wcss = []
max_clusters = 10
for k in range(1, max_clusters + 1):
    agglomerative = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agglomerative.fit_predict(X_scaled)

    # Calculate WCSS for the clusters formed by Agglomerative Clustering
    centroids = []
    for cluster_id in np.unique(labels):
        cluster_points = X_scaled[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Calculate WCSS
    _, dists = pairwise_distances_argmin_min(X_scaled, centroids)
    wcss.append(np.sum(dists ** 2))

# Step 3: Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal K in Agglomerative Clustering on Breast Cancer Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Define optimal k (from previous Elbow Method) and accuracy list
optimal_clusters = 3
accuracies = []

# List of linkage methods to try
linkage_methods = ['ward', 'complete', 'average', 'single']

# Apply hierarchical clustering and plot results for each linkage method
for method in linkage_methods:
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage=method)
    y_predicted = clustering.fit_predict(X_scaled)
    print(f"Linkage: {method}, prediction: {y_predicted}")

    # Accuracy
    accuracy = accuracy_score(y_true, y_predicted)
    accuracies.append((method, accuracy))
    print(f"Accuracy with {method} prediction: {accuracy:.4f}")

    # Plot dendrogram
    plt.figure(figsize=(8, 5))
    Z = linkage(X_scaled, method=method)
    dendrogram(Z)
    plt.title(f'Dendrogram ({method.capitalize()} Linkage)')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()

    # Plot clusters in PCA-reduced space
    plt.figure(figsize=(8, 5))
    for cluster in np.unique(y_predicted):
        plt.scatter(
            X_pca[y_predicted == cluster, 0], X_pca[y_predicted == cluster, 1],
            label=f'Cluster {cluster}', s=50, edgecolor='k'
        )
    plt.title(f'Hierarchical Clustering ({method.capitalize()} Linkage)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

for accuracy in accuracies:
    print(accuracy)
