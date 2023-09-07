import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# ---------------------Generate Sample Data----------------------
X, labels_true = make_blobs(n_samples=1000, cluster_std=0.5, random_state=0)

X = StandardScaler().fit_transform(X)

plt.figure(figsize=(12,4))
plt.subplot(121)

plt.plot(X[:, 0], X[:, 1],'b.')

plt.title("Original data")


# ------------------------Compute DBSCAN-------------------------
db = DBSCAN(eps=0.3, min_samples=5).fit(X)
# the maximum distance between two samples (eps) is 0.3, and
# the minimum number of samples in a neighbourhood for a point to be considered as a core point is 5
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


# -------------------------Plot Result---------------------------
plt.subplot(122)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()