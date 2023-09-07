import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# -----------Generate Sample Data & Initial Plotting-------------
plt.figure(figsize=(12,4))
X, y = make_blobs(n_samples=1500, cluster_std=0.5, random_state=0)
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Original data")


# ----------------------KMeans Clustering------------------------
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X)
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Clustered data")
plt.show()
