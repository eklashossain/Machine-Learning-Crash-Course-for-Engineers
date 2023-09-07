# ----------------------Importing Libraries----------------------
from sklearn import datasets
import pandas as pd
from scipy.stats import randint as sp_rand
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV
from datetime import datetime


# ------------------------Loading Dataset------------------------
# Downloading the Breast Cancer Dataset from Scikit-learn Library
dataset = datasets.load_breast_cancer()
data = pd.DataFrame(dataset.data, columns = dataset.feature_names)
data['target'] = dataset.target
data.head()

# Creating dataset variables
X = dataset.data
y = dataset.target

# Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------Hyperparameter Space----------------------
params_1 = {'n_estimators': 10, 'criterion': 'entropy',
            'max_features': 15, 'min_samples_split': 6,
            'min_samples_leaf': 8, 'bootstrap': True}
params_2 = {'n_estimators': 50, 'criterion': 'entropy',
            'max_features': 30, 'min_samples_split': 8,
            'min_samples_leaf': 11, 'bootstrap': True}
params_3 = {'n_estimators': 80, 'criterion': 'gini',
            'max_features': 30, 'min_samples_split': 10,
            'min_samples_leaf': 6, 'bootstrap': False}


# ---------------------Create and Fit Model----------------------
model_1 = RandomForestClassifier(**params_1)
model_2 = RandomForestClassifier(**params_2)
model_3 = RandomForestClassifier(**params_3)

# Training Models
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)

# Results
print(f'Model 1 accuracy: {model_1.score(X_test, y_test)}')
print(f'Model 2 accuracy: {model_2.score(X_test, y_test)}')
print(f'Model 3 accuracy: {model_3.score(X_test, y_test)}')


# ---------------------Hyperparameter Space----------------------
h_space = {'n_estimators': [30, 60, 80, 100],
           'criterion':['gini', 'entropy'],
           'max_features': [10, 20, 25, 30],
           'min_samples_split':[5, 11],
           'min_samples_leaf': [5, 11],
           'bootstrap': [True, False]}


# ---------------------Create and Fit Model----------------------
random_forest_clf = RandomForestClassifier()

# Creating and training models
# The datetime function has been used to calculate operation time
start = datetime.now()
models = GridSearchCV(random_forest_clf, param_grid = h_space, cv=5)
models.fit(X_train, y_train)
end = datetime.now()

# Getting 5-fold cross-validated score
scores = models.cv_results_['mean_test_score']
# Getting best hyperparameters
best_hparams = models.best_params_

print(f'Duration: {end-start}')
print(f'Best model training Score: {max(scores)}')
print(f'Best hyperparameters: {best_hparams}')


# ------------------------Training Model-------------------------
# Training model with best hyperparameters from Grid Search
best_model = RandomForestClassifier(bootstrap = True,
                                    criterion = 'entropy',
                                    max_features = 10,
                                    min_samples_leaf = 5,
                                    min_samples_split =  5,
                                    n_estimators = 80)
best_model.fit(X_train, y_train)
print('Best model accuracy: {best_model.score(X_test, y_test)}')


# ---------------------Create and Fit Model----------------------
# The hyperparameter space is the same as Grid Search
start = datetime.now()
# The datetime function has been used to calculate operation time
models = HalvingGridSearchCV(random_forest_clf,
                             param_grid = h_space, cv=5)
models.fit(X_train, y_train)
end = datetime.now()


# Getting 5-fold cross-validated score
scores = models.cv_results_['mean_test_score']
# Getting best hyperparameters
best_hparams = models.best_params_

print(f'Duration: {end-start}')
print(f'Best model training Score: {max(scores)}')
print(f'Best hyperparameters: {best_hparams}')


# ------------------------Training Model-------------------------
# Training model with best hyperparameters from Halving Grid Search
best_model = RandomForestClassifier(bootstrap = True,
                                    criterion = 'gini',
                                    max_features = 10,
                                    min_samples_leaf = 5,
                                    min_samples_split = 5,
                                    n_estimators = 30)
best_model.fit(X_train, y_train)
print(f'Best model accuracy: {best_model.score(X_test, y_test)}')


# ---------------------Hyperparameter Space----------------------
h_space = {'bootstrap': [True, False],
           'criterion': ['gini', 'entropy'],
           'max_features': sp_rand(2, 30),
           'min_samples_split': sp_rand(2, 11),
           'min_samples_leaf': sp_rand(2, 11),
           'n_estimators': sp_rand(30, 100)}


# ---------------------Create and Fit Model----------------------
start = datetime.now()
models = RandomizedSearchCV(random_forest_clf, param_distributions = h_space, cv=5,
                            random_state = 42)
models.fit(X_train, y_train)
end = datetime.now()

# Getting 5-fold cross-validated score
scores = models.cv_results_['mean_test_score']
# Getting best hyperparameters
best_hparams = models.best_params_

print(f'Duration: {end-start}')
print(f'Best model training Score: {max(scores)}')
print(f'Best hyperparameters: {best_hparams}')


# ------------------------Training Model-------------------------
# Training model with best hyperparameters from Random Search
best_model = RandomForestClassifier(bootstrap = True,
                                    criterion = 'entropy',
                                    max_features = 16,
                                    min_samples_leaf = 9,
                                    min_samples_split = 6,
                                    n_estimators = 53)
best_model.fit(X_train, y_train)
print(f'Best model accuracy: {best_model.score(X_test, y_test)}')


# ---------------------Create and Fit Model----------------------
# The hyperparameter space is the same as Random Search
start = datetime.now()
models = HalvingRandomSearchCV(random_forest_clf,
                               param_distributions = h_space,
                               cv=5,
                               random_state = 42)
models.fit(X_train, y_train)
end = datetime.now()

# Getting 5-fold cross-validated score
scores = models.cv_results_['mean_test_score']
# Getting best hyperparameters
best_hparams = models.best_params_

print(f'Duration: {end-start}')
print(f'Best model training Score: {max(scores)}')
print(f'Best hyperparameters: {best_hparams}')


# ------------------------Training Model-------------------------
# Training model with best hyperparameters from Halving Random Search
best_model = RandomForestClassifier(bootstrap = True,
                                    criterion = 'gini',
                                    max_features = 9,
                                    min_samples_leaf = 3,
                                    min_samples_split = 2,
                                    n_estimators = 77)
best_model.fit(X_train, y_train)
print(f'Best model accuracy: {best_model.score(X_test, y_test)}')