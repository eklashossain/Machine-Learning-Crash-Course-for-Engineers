#Dataset: https://www.kaggle.com/datasets/uciml/mushroom-classification
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------Reading Data--------------------------
df = pd.read_csv('./data/mushrooms.csv')


# ---------Encode These String Characters into Integers----------
encoder = LabelEncoder()


# Applying transformation
for column in df.columns:
    df[column] = encoder.fit_transform(df[column])

X = df.iloc[:,1:23]
Y = df.iloc[:, 0]


# ----------------------Scale the Features-----------------------
ss = StandardScaler()
X = ss.fit_transform(X)


# -------------------------Fit and Plot--------------------------
pca = PCA()
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_

# Plot before dimension reduction
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.bar(range(22), explained_variance, alpha=0.5, align='center', label='individual variance')
plt.title("Original data")


# Reduce the number of features using PCA
pca_reduced = PCA(n_components=15)
X_pca_reduced = pca_reduced.fit_transform(X)
explained_variance_reduced = pca_reduced.explained_variance_

# Plot after dimension reduction
plt.subplot(122)
plt.bar(range(15), explained_variance_reduced, alpha=0.5, align='center')
plt.title("Dimension Reduced")
plt.show()