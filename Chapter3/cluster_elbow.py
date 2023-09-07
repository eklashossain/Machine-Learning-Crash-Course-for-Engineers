import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# ---------------------Generate Sample Data----------------------
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)


# ---------Finding Out SSE Value for Number of Clusters----------
k_rng = range (1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)


# ---------------------------Plotting----------------------------
plt.xlabel('n')
plt.ylabel('sse')
plt.plot(k_rng,sse)
plt.show()