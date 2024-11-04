from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Load the dataset
wine = load_wine()
X = wine.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to identify the optimal number of components
pca = PCA()
pca.fit(X_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plotting the Cumulative Summation of the Explained Variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.grid(True)
plt.show()

# Select the top 2 principal components for visualization
pca_optimal = PCA(n_components=2)
X_pca = pca_optimal.fit_transform(X_scaled)

# Assess the quality of K-Means clusters using Within-Cluster Sum of Squares (WCSS)
# Parameters: init{‘k-means++’, ‘random’}, max_iter: int
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=600, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# The optimal number of clusters (k) - Elbow method
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o')
plt.title('The optimal number of clusters (k) - Elbow method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()

# Define optimal k (from previous Elbow Method)
optimal_k = 3

# Train KMeans on scaled data
kmeans_scaled = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=600, random_state=42)
y_kmeans_scaled = kmeans_scaled.fit_predict(X_scaled)

# Train KMeans on PCA-reduced data
kmeans_pca = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=600, random_state=42)
y_kmeans_pca = kmeans_pca.fit_predict(X_pca)

# Plot the clustering result for scaled data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for cluster in np.unique(y_kmeans_scaled):
    plt.scatter(
        X_scaled[y_kmeans_scaled == cluster, 0],
        X_scaled[y_kmeans_scaled == cluster, 1],
        label=f'Cluster {cluster}',
        s=50, edgecolor='k'
    )
plt.scatter(
    kmeans_scaled.cluster_centers_[:, 0],
    kmeans_scaled.cluster_centers_[:, 1],
    s=150, c='red', label='Centroids', edgecolor='k', marker='X'
)
plt.title(f'K-means Clustering (Scaled Data, k={optimal_k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Plot the clustering result for PCA-reduced data
plt.subplot(1, 2, 2)
for cluster in np.unique(y_kmeans_pca):
    plt.scatter(
        X_pca[y_kmeans_pca == cluster, 0],
        X_pca[y_kmeans_pca == cluster, 1],
        label=f'Cluster {cluster}',
        s=50, edgecolor='k'
    )
plt.scatter(
    kmeans_pca.cluster_centers_[:, 0],
    kmeans_pca.cluster_centers_[:, 1],
    s=150, c='red', label='Centroids', edgecolor='k', marker='X'
)
plt.title(f'K-means Clustering (PCA Data, k={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# # Scaled Optimal k (let's assume k=3 from visual inspection)
# optimal_k = 3
# kmeans_optimal = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=600, random_state=42)
# y_kmeans_optimal_s = kmeans_optimal.fit_predict(X_scaled)
#
# # Plot the clustering result for optimal k
# plt.figure(figsize=(8, 5))
# for cluster in np.unique(y_kmeans_optimal_s):
#     plt.scatter(
#         X_scaled[y_kmeans_optimal_s == cluster, 0],
#         X_scaled[y_kmeans_optimal_s == cluster, 1],
#         label=f'Cluster {cluster}',
#         edgecolor='k', s=50
#     )
# # Plot the centroids
# plt.scatter(
#     kmeans_optimal.cluster_centers_[:, 0],
#     kmeans_optimal.cluster_centers_[:, 1],
#     s=150, c='red', label='Centroids', edgecolor='k', marker='X'
# )
# plt.title(f'K-means Clustering (k={optimal_k})')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # PCA Optimal k (let's assume k=3 from visual inspection) -
# y_kmeans_optimal_p = kmeans_optimal.fit_predict(X_pca)
#
# # Plot the clustering result for optimal k
# plt.figure(figsize=(8, 5))
# for cluster in np.unique(y_kmeans_optimal_p):
#     plt.scatter(
#         X_scaled[y_kmeans_optimal_p == cluster, 0],
#         X_scaled[y_kmeans_optimal_p == cluster, 1],
#         label=f'Cluster {cluster}',
#         edgecolor='k', s=50
#     )
# # Plot the centroids
# plt.scatter(
#     kmeans_optimal.cluster_centers_[:, 0],
#     kmeans_optimal.cluster_centers_[:, 1],
#     s=150, c='red', label='Centroids', edgecolor='k', marker='X'
# )
# plt.title(f'K-means Clustering (k={optimal_k})')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.grid(True)
# plt.show()
