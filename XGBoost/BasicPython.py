

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:/AA - Naga/FreeTutorials.US/Udemy - machinelearning_Total/02 -------------------- Part 1 Data Preprocessing --------------------/Data_Preprocessing/Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

dataset.isnull().sum()

#Missing valuies imputation
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) 

##Creating dummy variables for country
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()

##Spliting dataset into training and testing
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)