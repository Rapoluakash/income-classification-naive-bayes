# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\rapol\Downloads\DATA SCIENCE\4. Dec\13th - NAIVE BAYES\adult.csv")



# Encode 'education' column (3rd column)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dataset['education'] = le.fit_transform(dataset['education'])


X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB() 
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


bias = classifier.score(X_train, y_train)
bias


#Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB() 
classifier1.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier1.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)
print(cm)

bias = classifier1.score(X_train, y_train)
bias

veriance = classifier1.score(X_train, y_train)
veriance

#Bernoulli Naive Bayes model
from sklearn.naive_bayes import BernoulliNB
classifier2 = BernoulliNB() 
classifier2.fit(X_train, y_train)


y_pred2 = classifier2.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred2)
print(ac)


bias = classifier2.score(X_train, y_train)
bias



verince = classifier2.score(X_train, y_train)
verince