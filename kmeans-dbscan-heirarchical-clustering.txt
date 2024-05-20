import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load datasets
iris = load_iris()
wine = load_wine()

datasets = {'Iris': iris, 'Wine': wine}

# Function to evaluate clustering
def evaluate_clustering(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    return silhouette_avg, davies_bouldin

# Function to plot clusters
def plot_clusters(X, labels, title):
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis')
    plt.title(title)
    plt.show()

# Scaling data
scaler = StandardScaler()

# Clustering settings
kmeans_params = [3, 5, 8]
dbscan_params = {'eps': [0.3, 0.5, 0.7], 'min_samples': [5, 10]}
hierarchical_params = [3, 5, 8]

# Evaluate clustering algorithms on both datasets
for dataset_name, dataset in datasets.items():
    print(f"Results for {dataset_name} dataset:")
    X, y = dataset.data, dataset.target
    X_scaled = scaler.fit_transform(X)
    
    # KMeans
    for n_clusters in kmeans_params:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_avg, davies_bouldin = evaluate_clustering(X_scaled, labels)
        print(f"KMeans with {n_clusters} clusters - Silhouette Score: {silhouette_avg:.4f}, Davies-Bouldin Score: {davies_bouldin:.4f}")
        plot_clusters(X_scaled, labels, f"KMeans with {n_clusters} clusters on {dataset_name}")
    
    # DBSCAN
    for eps in dbscan_params['eps']:
        for min_samples in dbscan_params['min_samples']:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            if len(set(labels)) > 1:  # Ensure we have more than one cluster
                silhouette_avg, davies_bouldin = evaluate_clustering(X_scaled, labels)
                print(f"DBSCAN with eps={eps} and min_samples={min_samples} - Silhouette Score: {silhouette_avg:.4f}, Davies-Bouldin Score: {davies_bouldin:.4f}")
                plot_clusters(X_scaled, labels, f"DBSCAN with eps={eps} and min_samples={min_samples} on {dataset_name}")
    
    # Hierarchical Clustering
    for n_clusters in hierarchical_params:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(X_scaled)
        silhouette_avg, davies_bouldin = evaluate_clustering(X_scaled, labels)
        print(f"Hierarchical Clustering with {n_clusters} clusters - Silhouette Score: {silhouette_avg:.4f}, Davies-Bouldin Score: {davies_bouldin:.4f}")
        plot_clusters(X_scaled, labels, f"Hierarchical Clustering with {n_clusters} clusters on {dataset_name}")

    print()
