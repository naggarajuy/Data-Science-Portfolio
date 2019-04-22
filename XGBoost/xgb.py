
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:/AA - Naga/Analytics Path/Python/Algorithms - Python/XGBoost/Churn_Modelling.csv')

# Checking for missing values with each columns
dataset.isnull().sum()

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

#Encoding the categorical data to numerical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

# spain is not greater than germany or o < 2, so we need to onehotencode
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Removing 1st column to escape dummy variable trap or To overcome multicollinearity
X = X[:,1:]

# To check dimensions of dataset
X.shape

## Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

## feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting XGBoost to the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) 

#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X_train, y_train, cv=10)
accuracies.mean()
accuracies.std()