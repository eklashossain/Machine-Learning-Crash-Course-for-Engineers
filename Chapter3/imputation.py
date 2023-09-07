import numpy as np
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 5],
         [2, 10],
         [7, 35],
         [15, 75],
         [0, 0]])

X = [[1, 5],
     [2, 10],
     [np.nan, 2],
     [7, 35],
     [6, np.nan],
     [15, 75],
     [0, 0]]

print(imp.transform(X))