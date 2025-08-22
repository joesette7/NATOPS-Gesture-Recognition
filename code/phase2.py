import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

PHASE1_CSV = "../Results/phase1_output.csv" 
PHASE2_CSV = "../Results/phase2_output.csv"  
N_CLUSTERS = 6                             

# Plot elbow method to determine optimal number of clusters
def plot_elbow(X_scaled, max_k=30):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method: Optimal k for KMeans')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig("../Results/elbow_phase2.png")
    plt.close()

# Plot silhouette scores to assess cluster cohesion and separation
def plot_silhouette(X_scaled, max_k=30):
    scores = []
    k_values = range(2, max_k + 1)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker='o')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig("../Results/silhouette_scores_phase2.png")
    plt.close()

def main():
    df = pd.read_csv(PHASE1_CSV)

    # Select only sensor feature columns
    feats = [col for col in df.columns if col.startswith('fea')]
    X = df[feats]

    # Normalize feature values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Generate evaluation plots for choosing optimal k
    print("Generating Elbow and Silhouette plots to evaluate optimal k...")
    plot_elbow(X_scaled, max_k=30)
    plot_silhouette(X_scaled, max_k=30)

    # Run KMeans clustering on time-step vectors
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df['cluster'] = cluster_labels

    # Group by sample ID (sid) and calculate cluster ratios
    ratio_list = []
    for sid, group in df.groupby('sid'):
        steps = group.shape[0]
        counts = group['cluster'].value_counts().sort_index()
        ratios = (counts / steps).reindex(range(N_CLUSTERS), fill_value=0)

        entry = {
            'isTest': group['isTest'].iloc[0],
            'sid': sid,
            'class': group['class'].iloc[0]
        }
        for i in range(N_CLUSTERS):
            entry[f'cluster{i}_ratio'] = ratios[i]
        ratio_list.append(entry)

    # Save new gesture-level features (cluster ratios)
    ratios_df = pd.DataFrame(ratio_list)
    ratios_df.to_csv(PHASE2_CSV, index=False)
    print(ratios_df.head())

    # Visualize clusters using PCA projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=10)
    plt.title('PCA Projection of Time Steps Colored by Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.savefig("../Results/phase2_clusters.png")

if __name__ == "__main__":
    main()

