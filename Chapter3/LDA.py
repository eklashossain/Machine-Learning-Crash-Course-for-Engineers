#Dataset: https://www.kaggle.com/competitions/titanic/data?select=train.csv
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -------------------------Reading Data--------------------------
df = pd.read_csv("./data/train.csv")
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna("S", inplace=True)
encoder = LabelEncoder()


# ------------------------Label Encoding-------------------------
encoder.fit(df["Sex"])



df_temp = encoder.transform(df["Sex"])
df["Sex"] = df_temp
encoder.fit(df["Embarked"])
df_temp = encoder.transform(df["Embarked"])
df["Embarked"] = df_temp


# Reshape data
agesArray = np.array(df["Age"]).reshape(-1, 1)
faresArray = np.array(df["Fare"]).reshape(-1, 1)


# Scale the X
ss = StandardScaler()
df["Age"] = ss.fit_transform(agesArray)
df["Fare"] = ss.fit_transform(faresArray)


# -------------------------Split Dataset-------------------------
X = df.drop(labels=['PassengerId', 'Survived'], axis=1)
Y = df['Survived']


# Splitting & fitting train data
xtrain, xval, ytrain, yval = train_test_split(X, Y, test_size=0.2, random_state=27)

lda_model = LDA()
lda_model.fit(xtrain, ytrain)
lda_predictions = lda_model.predict(xval)
lda_acc = accuracy_score(yval, lda_predictions)
lda_f1 = f1_score(yval, lda_predictions)

print("LDA Model - Accuracy: {}".format(lda_acc))
print("LDA Model - F1 Score: {}".format(lda_f1))


# ----------------------LDA Transformation-----------------------
lda_new = LDA(n_components=1)
lda_new.fit(X, Y)
X_lda = lda_new.transform(X)


# Printing result
print('Original feature #:', X.shape[1])
print('Reduced feature #:', X_lda.shape[1])


# Splitting with the new features and run the classifier
x_train_lda, x_val_lda, y_train_lda, y_val_lda = train_test_split(X_lda, Y, test_size=0.2, random_state=27)

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train_lda, y_train_lda)
logreg_predictions = logistic_regression.predict(x_val_lda)
logreg_acc = accuracy_score(y_val_lda, logreg_predictions)
logreg_f1 = f1_score(y_val_lda, logreg_predictions)
print("Logistic Regression Model - Accuracy: {}".format(logreg_acc))
print("Logistic Regression Model - F1 Score: {}".format(logreg_f1))