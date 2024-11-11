import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, \
    confusion_matrix
from sklearn.cluster import KMeans
import pandas as pd

# Load the Iris dataset
data = load_digits()
X = data.data
y_true = data.target

# Range of cluster numbers to evaluate
n_clusters_range = range(2, 11)
overall_silhouette_scores = []

# Evaluate KMeans clustering for each n_clusters value
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # Calculate overall silhouette score for the current number of clusters
    overall_silhouette_score = silhouette_score(X, labels)
    overall_silhouette_scores.append(overall_silhouette_score)

    # Calculate silhouette scores for each sample
    silhouette_vals = silhouette_samples(X, labels)

    # Display silhouette scores by cluster
    silhouette_df = pd.DataFrame({'Cluster': labels, 'Silhouette Score': silhouette_vals})
    cluster_silhouette_scores = silhouette_df.groupby('Cluster')['Silhouette Score'].mean()

    print(f"Overall Silhouette Score for n_clusters={n_clusters}: {overall_silhouette_score:.4f}")
    print(f"Silhouette Scores for Each Cluster (n_clusters={n_clusters}):")
    print(cluster_silhouette_scores)
    print("\n" + "-" * 50 + "\n")

# Plot overall silhouette scores for each n_clusters value
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, overall_silhouette_scores, marker='o', color='b')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Overall Silhouette Score')
plt.grid(True)
plt.show()

# ---------- Set the optimal number of clusters (assuming 3 for example) ------------
optimal_k = 8

# Fit KMeans with the chosen number of clusters
kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42)
labels_opt = kmeans_opt.fit_predict(X)

# Calculate silhouette scores
overall_silhouette_score_opt = silhouette_score(X, labels_opt)
silhouette_vals_opt = silhouette_samples(X, labels_opt)

# Calculate ARI and NMI
ari_score = adjusted_rand_score(y_true, labels_opt)
nmi_score = normalized_mutual_info_score(y_true, labels_opt)

print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

# Create confusion matrix to visually compare clusters and actual labels
conf_matrix = confusion_matrix(y_true, labels_opt)

# Print overall silhouette score
print(f"Overall Silhouette Score: {overall_silhouette_score_opt:.4f}")

# Prepare silhouette DataFrame for analysis
silhouette_df = pd.DataFrame({
    'Cluster': labels_opt,
    'Silhouette Score': silhouette_vals_opt
})

# 1. Silhouette Plot (Silhouette Diagram)
plt.figure(figsize=(10, 6))
y_lower = 10
for i in range(optimal_k):
    # Aggregate the silhouette scores for samples belonging to cluster i
    ith_cluster_silhouette_values = silhouette_vals_opt[labels_opt == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    # Fill silhouette plot with the silhouette scores for each cluster
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10  # 10 for space between plots

plt.title('Silhouette Plot for Each Cluster')
plt.xlabel('Silhouette Score')
plt.ylabel('Cluster')
plt.axvline(x=overall_silhouette_score_opt, color="red", linestyle="--")
plt.show()

# 2. Histogram of Silhouette Scores
plt.figure(figsize=(8, 5))
plt.hist(silhouette_vals_opt, bins=20, color='skyblue', edgecolor='k')
plt.title('Distribution of Silhouette Scores')
plt.xlabel('Silhouette Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 3. Box Plot of Silhouette Scores by Cluster
fig, ax = plt.subplots(figsize=(8, 5))
silhouette_df.boxplot(column='Silhouette Score', by='Cluster', ax=ax, grid=False)
ax.set_title('Box Plot of Silhouette Scores by Cluster')
ax.set_xlabel('Cluster')
ax.set_ylabel('Silhouette Score')
plt.suptitle('')  # Remove the automatic title added by plt.suptitle with pandas boxplot
plt.show()

# 4. Mean Silhouette Score by Cluster
mean_silhouette_scores = silhouette_df.groupby('Cluster')['Silhouette Score'].mean()
plt.figure(figsize=(8, 5))
mean_silhouette_scores.plot(kind='bar', color='lightgreen', edgecolor='k')
plt.title('Mean Silhouette Score for Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Mean Silhouette Score')
plt.grid(True)
plt.show()

# Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix: Actual Labels vs Predicted Clusters')
plt.xlabel('Predicted Clusters')
plt.ylabel('Actual Labels')
plt.show()
