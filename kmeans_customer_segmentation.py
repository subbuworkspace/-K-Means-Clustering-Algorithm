"""
K-Means Clustering - Customer Segmentation

Dataset:
    Mall_Customers.csv

Features:
    Annual Income (k$)
    Spending Score (1-100)

Author:
    Subrata Mondal
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# ============================================================
# 2. LOAD DATASET
# ============================================================


dataset = pd.read_csv(
    r"D:\work\olama resm\ML work\K-Means Clustering\Mall_Customers.csv"
)



# ============================================================
# 3. SELECT FEATURES
# ============================================================

# Column 3 → Annual Income (k$)
# Column 4 → Spending Score (1-100)

X = dataset.iloc[:, [3, 4]].values

print("Dataset Shape:", X.shape)


# ============================================================
# 4. ELBOW METHOD
# ============================================================

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=0,
        n_init=10
    )

    kmeans.fit(X)

    # WCSS = Within-Cluster Sum of Squares
    wcss.append(kmeans.inertia_)


# ============================================================
# 5. PLOT ELBOW METHOD
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.title("The Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid(True)

plt.show()


# ============================================================
# 6. CREATE FINAL K-MEANS MODEL
# ============================================================

# Based on the elbow analysis, we use K = 5

kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=0,
    n_init=10
)


# ============================================================
# 7. FIT MODEL AND ASSIGN CLUSTERS
# ============================================================

y_kmeans = kmeans.fit_predict(X)


# ============================================================
# 8. PRINT CLUSTER INFORMATION
# ============================================================

print("\nCluster Labels:")
print(y_kmeans)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)


# ============================================================
# 9. VISUALIZE CUSTOMER CLUSTERS
# ============================================================

plt.figure(figsize=(10, 7))

# Cluster 1
plt.scatter(
    X[y_kmeans == 0, 0],
    X[y_kmeans == 0, 1],
    s=100,
    c="red",
    label="Cluster 1"
)

# Cluster 2
plt.scatter(
    X[y_kmeans == 1, 0],
    X[y_kmeans == 1, 1],
    s=100,
    c="blue",
    label="Cluster 2"
)

# Cluster 3
plt.scatter(
    X[y_kmeans == 2, 0],
    X[y_kmeans == 2, 1],
    s=100,
    c="green",
    label="Cluster 3"
)

# Cluster 4
plt.scatter(
    X[y_kmeans == 3, 0],
    X[y_kmeans == 3, 1],
    s=100,
    c="cyan",
    label="Cluster 4"
)

# Cluster 5
plt.scatter(
    X[y_kmeans == 4, 0],
    X[y_kmeans == 4, 1],
    s=100,
    c="magenta",
    label="Cluster 5"
)


# ============================================================
# 10. PLOT CENTROIDS
# ============================================================

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="yellow",
    marker="X",
    edgecolor="black",
    label="Centroids"
)


# ============================================================
# 11. ADD LABELS
# ============================================================

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 12. DISPLAY NUMBER OF CUSTOMERS IN EACH CLUSTER
# ============================================================

print("\nCustomers in Each Cluster:")

for cluster in range(5):

    count = np.sum(y_kmeans == cluster)

    print(
        f"Cluster {cluster + 1}: {count} customers"
    )