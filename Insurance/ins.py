
# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import auc,roc_curve
from sklearn.model_selection import KFold
import time

#Importing Dataset
data = pd.read_csv('D:/Analytics Path/Capstone Projects/Other Projects/Insurance/insurance.csv')
d = data[1:3] # Fetching rows from DF
data.iloc[1:3,]

d1 = data.iloc[0:3, 1:4] # 3 rows and 3 columns
# Drop rows with missing values
#dataset.dropna(how='any',inplace=True)  # any of the column is naN that row is deleted [how= 'all' if all columns are naN]
# Fill missing values with mean column values
#dataset.fillna(dataset.mean(), inplace=True)

data.columns
data.head()
data.describe()
data.info()
data.dtypes()
data['insuranceclaim'].value_counts()
data['region'].value_counts()
data['insuranceclaim'] = data['insuranceclaim'].astype('category')
data['smoker'] = data['smoker'].astype('category')
data['sex'] = data['sex'].astype('category')
data['region'] = data['region'].astype('category')

data.shape

#Checking missing values
data.isnull().sum()
data['sex'].isnull().sum()

X = data.drop('insuranceclaim', axis=1)
y = data['insuranceclaim']

#Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=123)

#K-Folds
kf = KFold(n_splits=5) 
for fold_n, (train_index, test_index) in enumerate(kf.split(X,y)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

#Test performance
y_pred_test = rf.predict(X_test)
confusion_matrix(y_test, y_pred_test)
test_report = classification_report(y_test,y_pred_test)
acc = (110+143)/268

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_test)
auc(false_positive_rate, true_positive_rate)

#Train Performance
y_pred_train = rf.predict(X_train)
confusion_matrix(y_train, y_pred_train)
train_report = classification_report(y_train,y_pred_train)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_pred_train)
auc(false_positive_rate, true_positive_rate)

###### Date format Conversion ################
import datetime

datetime.datetime.strptime("2013-1-25", '%Y-%m-%d').strftime('%m/%d/%y')

#or

d = datetime.datetime.strptime("2013-1-25", '%Y-%m-%d')
print (datetime.datetime.strftime(d, '%m/%d/%y'))