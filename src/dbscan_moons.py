import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons

# Generate moon-shaped data
data, _ = make_moons(n_samples=600, noise=0.1, random_state=42)

# Apply DBSCAN and KMeans
dbscan = DBSCAN(eps=0.15, min_samples=10)
kmeans = KMeans(n_clusters=2)

# Get cluster labels
dbscan_clusters = dbscan.fit_predict(data)
kmeans_clusters = kmeans.fit_predict(data)

# Print cluster labels for inspection
print("DBSCAN Clusters:\n", dbscan_clusters)
print("KMeans Clusters:\n", kmeans_clusters)

# Exclude noise points (labeled as -1) for silhouette calculation
dbscan_mask = dbscan_clusters != -1
dbscan_filtered_data = data[dbscan_mask]
dbscan_filtered_labels = dbscan_clusters[dbscan_mask]

# Calculate silhouette scores
dbscan_silhouette_score = silhouette_score(dbscan_filtered_data, dbscan_filtered_labels)
kmeans_silhouette_score = silhouette_score(data, kmeans_clusters)

print(f"Overall Silhouette Score for DBSCAN (without noise): {dbscan_silhouette_score:.4f}")
print(f"Overall Silhouette Score for KMeans: {kmeans_silhouette_score:.4f}")

# Plot both DBSCAN and KMeans clustering results side-by-side
plt.figure(figsize=(16, 6))

# DBSCAN Plot
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=dbscan_clusters, cmap='plasma', marker='o', edgecolor='k')
plt.title("DBSCAN Clustering with Noise")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# KMeans Plot
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=kmeans_clusters, cmap='plasma', marker='o', edgecolor='k')
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
