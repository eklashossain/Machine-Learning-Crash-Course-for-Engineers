from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle


# -----------Generate Sample Data & Initial Plotting-------------
X, labels_true = make_blobs(n_samples=300, cluster_std=0.5, random_state=0)
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(X[:, 0], X[:, 1],'.')
plt.title("Original data")


# -----------------Compute Affinity Propagation------------------
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

cluster_num = len(cluster_centers_indices)


# -------------------------Plot Results--------------------------
plt.subplot(122)
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(cluster_num), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % cluster_num)
plt.show()