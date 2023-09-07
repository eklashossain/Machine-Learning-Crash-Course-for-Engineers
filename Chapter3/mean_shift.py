import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle


# ---------------------Generate Sample Data----------------------
X, _ = make_blobs(n_samples=10000,cluster_std=0.5,random_state=0)


# ---------------------------Plotting----------------------------
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(X[:, 0], X[:, 1],'.')
plt.title("Original data")


# --------------Compute Clustering with MeanShift----------------
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


# -------------------------Plot Results--------------------------
plt.subplot(122)
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()