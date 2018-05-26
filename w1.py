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
    # data.info(verbose=False, memory_usage=False)

    # Subconjunto dos dados que vou utilizar
    data_analysis = data.drop(["latitude", "longitude"], axis=1)
    # data_analysis = data[["latitude", "longitude"]]

    # PCA
    centralized_data = data_analysis - data_analysis.mean()
    pca = PCA(n_components=len(centralized_data.columns))
    data_pca = pca.fit_transform(centralized_data)

    df_pca = pd.DataFrame(data_pca)

    variance = pca.explained_variance_ratio_
    for i, j in enumerate(variance):
        variance[i] = sum(variance[i - 1:i + 1]) if i > 0 else variance[i]

    # 22dim => 0.75 variance
    # 46dim => 0.9 variance
    df_pca = df_pca.drop(
        [column for column in range(22, len(df_pca.columns))], axis=1)

    data_analysis = df_pca

    data_analysis.insert(len(data_analysis.columns),
                         "latitude", data["latitude"])
    data_analysis.insert(len(data_analysis.columns),
                         "longitude", data["longitude"])

    # Normalizando os dados
    for column in data_analysis:
        data_analysis[column] = (data_analysis[column] - data_analysis[column].min()) / \
            (data_analysis[column].max() - data_analysis[column].min())

    # Roubando...
    data_analysis["latitude"] *= 5.5
    data_analysis["longitude"] *= 5.5
    # Achar k com inércia
    '''
    inertia = []
    for k in range(2, 100):
        km = KMeans(n_clusters=k, n_jobs=-1)
        labels = km.fit_predict(data_analysis)
        inertia.append(km.inertia_)
    sns.set(font_scale=2)
    plt.plot(range(2, 100), inertia)
    plt.title("Valor da inércia variando o número de grupos")
    plt.xlabel("Número de grupos")
    plt.ylabel("Inércia")
    plt.show()
    '''
    # valores de k só para latitude e longitude => 7-11

    # Achar k com silhueta

    # 2, k=11
    # 2.3, k=15,14

    # spectral: 3.5, k=15
    # aglomerative: 5, k=15
    n_clusters = 15
    #clusterer = KMeans(n_clusters=n_clusters)
    clusterer = KMeans(n_clusters=n_clusters, n_jobs=-1)
    cluster_labels = clusterer.fit_predict(data_analysis)

    data.insert(len(data.columns),
                "label", cluster_labels)

    all_categories = []
    for i in range(n_clusters):
        data_cluster = data[data.label == i]
        data_cluster = data_cluster.drop(
            ["latitude", "longitude", "label"], axis=1)

        # print(data_cluster.sum().sort_values(ascending=False)[:3])
        # print()
        all_categories.append(
            data_cluster.sum().sort_values(ascending=False)[1:4])

    # Verifica se existe trios de categorias repetidos
    for i in range(len(all_categories)):
        for j in range(i + 1, len(all_categories)):
            if all_categories[i].index[0] in all_categories[j].index and all_categories[i].index[1] in all_categories[j].index \
                    and all_categories[i].index[2] in all_categories[j].index:
                for category in all_categories[i].index:
                    print(category, end=" ")
                print()
                for category in all_categories[j].index:
                    print(category, end=" ")
                print()
                #print(all_categories[i].index, all_categories[j].index)
                

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data_analysis) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(data_analysis, cluster_labels)
    print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(
        data_analysis, cluster_labels)

    y_lower = 10

    colors_vec = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8",
                    "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#fabebe", "#008080",
                    "#e6beff", "#800000", "#000080", "#000000", "#d2f53c", "#808080"]

    color = sns.color_palette(colors_vec)
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
    ax2.scatter(data["latitude"].as_matrix(), data["longitude"].as_matrix(), marker='o', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')

    plt.show()


if __name__ == "__main__":
    main()
