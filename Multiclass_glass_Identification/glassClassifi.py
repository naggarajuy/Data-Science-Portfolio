
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Importing dataset
glass = pd.read_csv('D:/Analytics Path/Capstone Projects/Other Projects/uci-glass-identification/glass.data.csv')

glass.columns
glass.head()
glass.info()
glass.shape
glass['Type of glass'].value_counts()

#Checking missing values
glass.isnull().sum()

#Spliting the data
X = glass.iloc[:,1:9].values
y = glass.iloc[:,10].values

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20, random_state=0)

#Logistic Regression
log_cls = LogisticRegression()
log_cls.fit(X_train,y_train)

y_pred = log_cls.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))

plt.figure(figsize = (6,6))
sns.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Actual')


##Decision tree classifier
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
dtree_cm = confusion_matrix(y_test, dtree_predictions) 
acc = (7+8+5)/43

#SVM
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
# creating a confusion matrix 
svm_cm = confusion_matrix(y_test, svm_predictions) 
acc = (8+7+1+6)/43

#KNN
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)   
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
knn_cm = confusion_matrix(y_test, knn_predictions) 
acc = (6+12+1+1+5)/43

#Naive Bayes
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test)  
# creating a confusion matrix 
naive_cm = confusion_matrix(y_test, gnb_predictions) 

#Keras
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)