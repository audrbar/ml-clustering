from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
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

# Initialize and apply the StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(df[['Spending Score (1-100)', 'Annual Income (k$)']])

# Define parameter ranges
dbscan_eps_values = [0.2, 0.3, 0.4, 0.5]
dbscan_min_samples_values = [3, 5, 10]

# Initialize lists to store silhouette and Davies-Bouldin scores
dbscan_scores = []

# Explore DBSCAN parameters and calculate silhouette and Davies-Bouldin scores
for eps in dbscan_eps_values:
    for min_samples in dbscan_min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_clusters = dbscan.fit_predict(data)

        # Calculate the percentage of noise points (-1 labels)
        noise_ratio = np.sum(dbscan_clusters == -1) / len(dbscan_clusters)

        # Exclude results with more than 50% noise points
        if noise_ratio < 0.5:
            # Filter out noise points for silhouette and Davies-Bouldin scores
            dbscan_mask = dbscan_clusters != -1
            dbscan_filtered_data = data[dbscan_mask]
            dbscan_filtered_labels = dbscan_clusters[dbscan_mask]

            # Calculate silhouette and Davies-Bouldin scores
            if len(np.unique(dbscan_filtered_labels)) > 1:
                silhouette = silhouette_score(dbscan_filtered_data, dbscan_filtered_labels)
                davies_bouldin = davies_bouldin_score(dbscan_filtered_data, dbscan_filtered_labels)
                dbscan_scores.append((eps, min_samples, silhouette, davies_bouldin))
            else:
                dbscan_scores.append((eps, min_samples, np.nan, np.nan))

# Convert scores to a DataFrame for plotting
dbscan_scores_df = pd.DataFrame(dbscan_scores, columns=['eps', 'min_samples', 'Silhouette Score', 'Davies-Bouldin Score'])

# Find best parameter combinations for each metric
best_silhouette = dbscan_scores_df.loc[dbscan_scores_df['Silhouette Score'].idxmax()]
best_davies_bouldin = dbscan_scores_df.loc[dbscan_scores_df['Davies-Bouldin Score'].idxmin()]

# Print best parameter combinations for reference
print("Best Silhouette Score parameters:\n", best_silhouette[['eps', 'min_samples', 'Silhouette Score']])
print("Best Davies-Bouldin Score parameters:\n", best_davies_bouldin[['eps', 'min_samples', 'Davies-Bouldin Score']])

# Plot comparison of Silhouette and Davies-Bouldin scores for each parameter combination
plt.figure(figsize=(14, 8))

for eps in dbscan_eps_values:
    subset = dbscan_scores_df[dbscan_scores_df['eps'] == eps]
    plt.plot(subset['min_samples'], subset['Silhouette Score'], label=f'Silhouette (eps={eps})', marker='o', linestyle='--')
    plt.plot(subset['min_samples'], subset['Davies-Bouldin Score'], label=f'Davies-Bouldin (eps={eps})', marker='s', linestyle='-')

# Highlight best Silhouette Score
plt.scatter(best_silhouette['min_samples'], best_silhouette['Silhouette Score'], color='red', s=100, label='Best Silhouette Score', marker='*')

# Highlight best Davies-Bouldin Score
plt.scatter(best_davies_bouldin['min_samples'], best_davies_bouldin['Davies-Bouldin Score'], color='blue', s=100, label='Best Davies-Bouldin Score', marker='*')

plt.xlabel("min_samples (Minimum Samples)")
plt.ylabel("Score")
plt.title("Comparison of Silhouette Score and Davies-Bouldin Score for DBSCAN")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
