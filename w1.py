import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np


def main():
    # Lendo os dados
    data = pd.read_csv("data/data_pt1.csv")
    data.info(verbose=False, memory_usage=False)

    # Normalizando os dados
    for column in data:
        data[column] = (data[column] - data[column].min()) / \
            (data[column].max() - data[column].min())

    # Subconjunto dos dados que vou utilizar
    #data_analysis = data[["latitude", "longitude"]]
    data_analysis = data.drop(["latitude", "longitude"], axis=1)

    # PCA
    centralized_data = data_analysis - data_analysis.mean()
    pca = PCA(n_components=len(data_analysis.columns))
    data_pca = pca.fit_transform(centralized_data)

    df_pca = pd.DataFrame(data_pca)

    variance = pca.explained_variance_ratio_
    for i, j in enumerate(variance):
        variance[i] = sum(variance[i - 1:i + 1]) if i > 0 else variance[i]

    # 30 dim => 0.82 variancia

    df_pca = df_pca.drop(
        [column for column in range(30, len(df_pca.columns))], axis=1)

    data_analysis = df_pca
    # data_analysis.insert(len(data_analysis.column),
    #                     latitude, data["longitude"])
    # data_analysis.insert(len(data_analysis.column),
    #                     longitude, data["longitude"])

    # print(data_analysis.head())
    # return
    # Achar k com inércia
    '''
    inertia = []
    for k in range(2, 100):
        km = KMeans(n_clusters=k)
        labels = km.fit_predict(data_analysis)
        inertia.append(km.inertia_)
    plt.plot(range(2, 100), inertia)
    plt.show()

    return
    '''
    # Achar k com silhueta
    range_n_clusters = range(10, 30)

    # Valores de latitude e longitude em numpy
    longitude = data["longitude"].as_matrix()
    latitude = data["latitude"].as_matrix()

    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(data_analysis) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(data_analysis)

        silhouette_avg = silhouette_score(data_analysis, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        sample_silhouette_values = silhouette_samples(
            data_analysis, cluster_labels)

        y_lower = 10

        color = (sns.color_palette("husl", n_clusters))
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color[i], edgecolor=color[i], alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = [color[i] for i in cluster_labels]
        ax2.scatter(latitude, longitude, marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()


if __name__ == "__main__":
    main()
