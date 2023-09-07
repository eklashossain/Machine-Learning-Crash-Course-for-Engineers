# --------------------Importing Libraries------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_gaussian_quantiles


# ----------Defining classifier models and datasets--------------
names = ["KNN", "SVM", "RBF SVM", "Gaussian Process",
         "Gaussian NB", "Decision Tree", "Random Forest"]

# 6 classifiers are demonstrated here
models = [
    KNeighborsClassifier(3),  # KNN
    SVC(kernel="linear", C=0.02), # SVM
    SVC(gamma=2, C=1), # Non-Linear SVM
    GaussianProcessClassifier(0.99 * RBF(1.0)), # Gaussian classifier
    GaussianNB(), # NB
    DecisionTreeClassifier(max_depth=6), # Decision tree
    RandomForestClassifier(max_depth=6, n_estimators=10, max_features=2), # Random Forest
    ]


datasets = [make_moons(noise=0.1, random_state=1),
            # creates a moon shape 2 class dataset

            make_classification(n_features=2, n_redundant=0,
                                n_informative=1,
                                n_clusters_per_class=1),
            # creates a separable classification dataset

            make_gaussian_quantiles(n_features=2, n_classes=2),
            # creates two gaussian circle dataset

            make_classification(n_features=2, n_redundant=0,
                                n_informative=2)
            # creates a dataset with two class overlap
            ]


Fig = plt.figure(figsize=(30, 10))
count = 1


# -------------------Training and Plotting-----------------------
for idxx, data in enumerate(datasets):
    X, y = data
    # pre-processing
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.5, random_state=10)

    # Minimum and maximum range for creating the plot mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))


    # Using the scikit-learn plotting of decision boundary example
    cm = plt.cm.gray
    cm_bright = ListedColormap(['#F000A0', '#00FFAA'])
    ax = plt.subplot(len(datasets), len(models) + 1, count)
    if idxx == 0:
        ax.set_title("Different types of data")
    
    # plotting test samples
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    count += 1 # Counting column

    
    for name, classifier in zip(names, models):
        ax = plt.subplot(len(datasets), len(models) + 1, count)
        classifier.fit(X_train, y_train) # Training
        acc = classifier.score(X_test, y_test) *100 # Accuracy
 
        # Decision boundary plotting
        # Point in the mesh [x_min, x_max]x[y_min, y_max]
        if hasattr(classifier, "decision_function"):
            hh = classifier.decision_function(np.c_[xx.ravel(),
                                                    yy.ravel()])
        else:
            hh = classifier.predict_proba(np.c_[xx.ravel(),
                                                yy.ravel()])[:,1]

        # Put the result into a color plot
        hh = hh.reshape(xx.shape)
        ax.contourf(xx, yy, hh,cmap=cm, alpha=.8)

       
        # Ploting test samples
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.8)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if idxx == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % acc).lstrip('0'),
                size=15, color= 'red', horizontalalignment='right')
        count += 1 # counting within each row


plt.savefig('./results/classifier.png', bbox_inches='tight')