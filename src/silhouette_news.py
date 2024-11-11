from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import pandas as pd

# Load the Iris dataset
data = fetch_20newsgroups()
X = data.data
y = data.target

# Preprocess the text data using TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
X_tfidf = tfidf.fit_transform(X)

print(f"\nBagofWords Length: {len(tfidf.get_feature_names_out())}")
print(f"X_tfidf Length: {len(X_tfidf.toarray())}")
print(f"\nBagofWords[5350:5360]: \n{tfidf.get_feature_names_out()[5350:5360]}")
print(f"X_tfidf[500:515]: \n{X_tfidf.toarray()[500:515]}")
print(f"Target: \n{y}")

# Range of cluster numbers to evaluate
n_clusters_range = range(10, 20)
overall_silhouette_scores = []

# Evaluate KMeans clustering for each n_clusters value
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_tfidf)

    # Calculate overall silhouette score for the current number of clusters
    overall_silhouette_score = silhouette_score(X_tfidf, labels)
    overall_silhouette_scores.append(overall_silhouette_score)

    # Calculate silhouette scores for each sample
    silhouette_vals = silhouette_samples(X_tfidf, labels)

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
