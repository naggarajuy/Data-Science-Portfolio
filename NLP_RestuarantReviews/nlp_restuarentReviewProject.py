# Natural Language Processing

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv('D:/Analytics Path/Capstone Projects/My Projects/NLP_RestuarentReviews/Datasets/Restaurant_Reviews.tsv',
                      delimiter ='\t', quoting = 3)
 
#Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Sparse matrix will consider each word as a column i.e the reason we do stemming to avoid sparsity

#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
#BagOfWords is CountVectorizer created each column for each word
# In CountVectorizer we can also do Stopwords, lowercase, max_features steps but it will
#not be in our countrol
# Better to do seperately
# By using max_features parameter we can use most frequent words for occured in review
cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.20, random_state= 0)

#Fitting the Naive Bayes to training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Accuracy
(55+91)/200

#Trying better performance with RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
