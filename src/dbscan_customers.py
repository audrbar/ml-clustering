from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Data
pd.options.display.max_columns = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-clustering/data/Mall_Customers.csv')

# Drop some columns
df = df.drop(df.columns[[0]], axis=1)

# Handle missing values (if any)
df = df.dropna()  # Or: df = df.fillna(df.mean())

# Apply LabelEncoder on Categorical columns:
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Select numeric columns (int64 and float64)
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# Filter numeric columns where the maximum value is greater than 10
filtered_columns = numeric_columns.loc[:, numeric_columns.max() > 10]
print(f"\nDF Filtered Columns: \n{filtered_columns.columns}")

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling only to the filtered columns (those where max > 10)
df[filtered_columns.columns] = scaler.fit_transform(df[filtered_columns.columns])
print(f"\nDF Describe after normalization: \n{df.describe()}")

# Explore Data
print(df.info())
print("\nUniques:")
print(f"Gender: {df['Gender'].unique()}")
print(f"Age: {df['Age'].unique()}")
print(f"Annual Income: {df['Annual Income (k$)'].unique()}")
print(f"Spending Score: {df['Spending Score (1-100)'].unique()}")

# grade_mapping = { 1: 'A', 2: 'B', 3: 'C' }
# df['letter_grade'] = df['grade'].map(grade_mapping)
# df['status'] = df['status'].replace({'fail': 'F', 'pass': 'P'})
# Reduce to 2 dimensions for visualization
# pca = PCA(n_components=2)
# data = pca.fit_transform(initial_data)

# Generate data
all_data = df[['Gender', 'Age', 'Spending Score (1-100)', 'Annual Income (k$)']]
some_data = df[['Spending Score (1-100)', 'Annual Income (k$)']]
print(f"Spending data: {all_data.columns}")
print(f"Income data: {some_data.columns}")

data = some_data

# Define parameter ranges
dbscan_eps_values = [0.2, 0.3, 0.4, 0.5]
dbscan_min_samples_values = [3, 5, 10]
kmeans_cluster_values = [3, 5, 7]

# Initialize lists to store silhouette scores
dbscan_silhouette_scores = []
kmeans_silhouette_scores = []
filtered_dbscan_silhouette_scores = []

# Explore DBSCAN parameters and calculate silhouette scores
for eps in dbscan_eps_values:
    for min_samples in dbscan_min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_clusters = dbscan.fit_predict(data)

        # Calculate the percentage of noise points (-1 labels)
        noise_ratio = np.sum(dbscan_clusters == -1) / len(dbscan_clusters)

        # Exclude results with more than 50% noise points
        if noise_ratio < 0.5:
            # Filter out noise points for silhouette score
            dbscan_mask = dbscan_clusters != -1
            dbscan_filtered_data = data[dbscan_mask]
            dbscan_filtered_labels = dbscan_clusters[dbscan_mask]

            # Calculate silhouette score if there are clusters formed
            if len(np.unique(dbscan_filtered_labels)) > 1:
                dbscan_score = silhouette_score(dbscan_filtered_data, dbscan_filtered_labels)
                filtered_dbscan_silhouette_scores.append((eps, min_samples, dbscan_score, noise_ratio))
            else:
                dbscan_score = np.nan
                filtered_dbscan_silhouette_scores.append((eps, min_samples, dbscan_score, noise_ratio))

        # Plot DBSCAN clustering result
        plt.figure(figsize=(10, 6))
        plt.scatter(data[['Spending Score (1-100)']], data[['Annual Income (k$)']], c=dbscan_clusters, cmap='plasma', marker='o', edgecolor='k')
        plt.title(f"DBSCAN (eps={eps}, min_samples={min_samples})\nSilhouette Score: {dbscan_score:.4f}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

# Find the best DBSCAN parameters based on the highest silhouette score (from filtered results)
if filtered_dbscan_silhouette_scores:
    best_dbscan_params = max(filtered_dbscan_silhouette_scores, key=lambda x: x[2])
    print(f"Best DBSCAN Parameters: eps={best_dbscan_params[0]}, min_samples={best_dbscan_params[1]}")
    print(f"Best DBSCAN Silhouette Score: {best_dbscan_params[2]:.4f}")
    print(f"Noise Ratio with Best Params: {best_dbscan_params[3]:.2%}")

    # Make best DBSCAN clustering result
    best_dbscan = DBSCAN(eps=best_dbscan_params[0], min_samples=best_dbscan_params[1])
    best_dbscan_clusters = best_dbscan.fit_predict(data)
else:
    print("No suitable DBSCAN parameters found with <50% noise.")
    best_dbscan_params = None

# Explore KMeans parameters and calculate silhouette scores
for n_clusters in kmeans_cluster_values:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(data)

    # Calculate silhouette score for KMeans
    kmeans_score = silhouette_score(data, kmeans_clusters)
    kmeans_silhouette_scores.append((n_clusters, kmeans_score))

    # Plot KMeans clustering result
    plt.figure(figsize=(10, 6))
    plt.scatter(data[['Spending Score (1-100)']], data[['Annual Income (k$)']], c=kmeans_clusters, cmap='plasma', marker='o', edgecolor='k')
    plt.title(f"KMeans (n_clusters={n_clusters})\nSilhouette Score: {kmeans_score:.4f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Find best KMeans parameters based on the highest silhouette score
best_kmeans_params = max(kmeans_silhouette_scores, key=lambda x: x[1])
print(f"Best KMeans Parameters: n_clusters={best_kmeans_params[0]}")
print(f"Best KMeans Silhouette Score: {best_kmeans_params[1]:.4f}")

# Make best KMeans clustering result
best_kmeans = KMeans(n_clusters=best_kmeans_params[0], random_state=42)
best_kmeans_clusters = best_kmeans.fit_predict(data)

# Plot both best DBSCAN and KMeans clustering results side-by-side
plt.figure(figsize=(14, 6))

# DBSCAN subplot
plt.subplot(1, 2, 1)
plt.scatter(data[['Spending Score (1-100)']], data[['Annual Income (k$)']], c=best_dbscan_clusters, cmap='plasma', marker='o', edgecolor='k')
plt.title(f"Best DBSCAN (eps={best_dbscan_params[0]}, min_samples={best_dbscan_params[1]})\nSilhouette Score: {best_dbscan_params[2]:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# KMeans subplot
plt.subplot(1, 2, 2)
plt.scatter(data[['Spending Score (1-100)']], data[['Annual Income (k$)']], c=best_kmeans_clusters, cmap='plasma', marker='o', edgecolor='k')
plt.title(f"Best KMeans (n_clusters={best_kmeans_params[0]})\nSilhouette Score: {best_kmeans_params[1]:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
