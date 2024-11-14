from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load Data
pd.options.display.max_columns = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-clustering/data/Mall_Customers.csv')

# Drop some columns and handle missing values (if any)
df = df.drop(df.columns[[0]], axis=1)
df = df.dropna()  # Or: df = df.fillna(df.mean())


# Define Dunn Index calculation function using a distance matrix
def dunn_score(i_distance_matrix, labels):
    unique_clusters = np.unique(labels)
    unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude noise points

    if len(unique_clusters) < 2:
        return np.nan  # Dunn Index is undefined for fewer than 2 clusters

    # Calculate minimum inter-cluster distance
    inter_cluster_distances = []
    for i, cluster_a in enumerate(unique_clusters):
        for cluster_b in unique_clusters[i + 1:]:
            cluster_a_points = np.where(labels == cluster_a)[0]
            cluster_b_points = np.where(labels == cluster_b)[0]
            inter_cluster_distances.append(np.min(i_distance_matrix[cluster_a_points][:, cluster_b_points]))

    # Calculate maximum intra-cluster distance
    intra_cluster_distances = []
    for cluster in unique_clusters:
        cluster_points = np.where(labels == cluster)[0]
        if len(cluster_points) > 1:
            intra_cluster_distances.append(np.max(i_distance_matrix[cluster_points][:, cluster_points]))

    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)


# Initialize and apply the StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(df[['Spending Score (1-100)', 'Annual Income (k$)']])

# Define parameter ranges
dbscan_eps_values = [0.2, 0.3, 0.4, 0.5]
dbscan_min_samples_values = [3, 5, 10]

# Initialize lists to store clustering quality metrics
dbscan_scores = []

# Precompute the distance matrix for the data
distance_matrix = pairwise_distances(data)

# Explore DBSCAN parameters and calculate Silhouette, Davies-Bouldin, and Dunn scores
for eps in dbscan_eps_values:
    for min_samples in dbscan_min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_clusters = dbscan.fit_predict(data)

        # Calculate the percentage of noise points (-1 labels)
        noise_ratio = np.sum(dbscan_clusters == -1) / len(dbscan_clusters)

        # Exclude results with more than 50% noise points
        if noise_ratio < 0.5:
            # Filter out noise points for metrics calculation
            dbscan_mask = dbscan_clusters != -1
            dbscan_filtered_data = data[dbscan_mask]
            dbscan_filtered_labels = dbscan_clusters[dbscan_mask]

            # Calculate Silhouette, Davies-Bouldin, and Dunn Index scores
            if len(np.unique(dbscan_filtered_labels)) > 1:
                silhouette = silhouette_score(dbscan_filtered_data, dbscan_filtered_labels)
                davies_bouldin = davies_bouldin_score(dbscan_filtered_data, dbscan_filtered_labels)
                dunn_index_value = dunn_score(distance_matrix[dbscan_mask][:, dbscan_mask], dbscan_filtered_labels)
                dbscan_scores.append((eps, min_samples, silhouette, davies_bouldin, dunn_index_value))
            else:
                dbscan_scores.append((eps, min_samples, np.nan, np.nan, np.nan))

# Convert scores to a DataFrame for plotting
dbscan_scores_df = pd.DataFrame(
    dbscan_scores, columns=['eps', 'min_samples', 'Silhouette Score', 'Davies-Bouldin Score', 'Dunn Index'])

# Identify the best parameter combinations for each metric
best_silhouette = dbscan_scores_df.loc[dbscan_scores_df['Silhouette Score'].idxmax()]
best_davies_bouldin = dbscan_scores_df.loc[dbscan_scores_df['Davies-Bouldin Score'].idxmin()]
best_dunn = dbscan_scores_df.loc[dbscan_scores_df['Dunn Index'].idxmax()]

# Print the results for each (eps, min_samples) combination
print("DBSCAN Parameter Tuning Results:")
print(dbscan_scores_df.to_string(index=False))

# Print best parameter combinations for reference
print("\nBest Silhouette Score parameters:\n", best_silhouette[['eps', 'min_samples', 'Silhouette Score']])
print("\nBest Davies-Bouldin Score parameters:\n", best_davies_bouldin[['eps', 'min_samples', 'Davies-Bouldin Score']])
print("\nBest Dunn Index parameters:\n", best_dunn[['eps', 'min_samples', 'Dunn Index']])

# Plot comparison of Silhouette, Davies-Bouldin, and Dunn Index scores for each parameter combination
plt.figure(figsize=(14, 8))

for eps in dbscan_eps_values:
    subset = dbscan_scores_df[dbscan_scores_df['eps'] == eps]
    plt.plot(subset['min_samples'], subset['Silhouette Score'], label=f'Silhouette (eps={eps})',
             marker='o', linestyle='--')
    plt.plot(subset['min_samples'], subset['Davies-Bouldin Score'], label=f'Davies-Bouldin (eps={eps})',
             marker='s', linestyle='-')
    plt.plot(subset['min_samples'], subset['Dunn Index'], label=f'Dunn Index (eps={eps})',
             marker='^', linestyle='-.')

# Highlight the best scores
plt.scatter(best_silhouette['min_samples'], best_silhouette['Silhouette Score'], color='red', s=100,
            label='Best Silhouette Score', marker='*')
plt.scatter(best_davies_bouldin['min_samples'], best_davies_bouldin['Davies-Bouldin Score'], color='blue', s=100,
            label='Best Davies-Bouldin Score', marker='*')
plt.scatter(best_dunn['min_samples'], best_dunn['Dunn Index'], color='green', s=100,
            label='Best Dunn Index', marker='*')

# Labels and Title
plt.xlabel("min_samples (Minimum Samples)")
plt.ylabel("Score")
plt.title("Comparison of Silhouette, Davies-Bouldin, and Dunn Index for DBSCAN")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
