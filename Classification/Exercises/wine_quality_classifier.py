import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def show_confusion_matrix(sgd_clf, X_train, y_train):
    """
        The general idea is to count the number of times instances of class A are
        classified as class B. A perfect classifier would have only true positives and true
        negatives
        For example, to know the number of times the classifier confused
        images of 5s with 3s, you would look in the 5th row and 3rd column of the confusion
        matrix
    """
    y_train_previsti = cross_val_predict(sgd_clf, X_train, y_train, cv=3) # Restituisce un array di boolean per ogni X_train di risultati previsti
    # Matrix 2 x 2
    # [[True Positive, False Positive] 
    # [False Negative, True Negative]]
    matrix = confusion_matrix(y_train, y_train_previsti) 
    return matrix


data = pd.read_csv('winequality-white.csv', sep=';')
X = data.iloc[:, data.columns != 'quality']
y = data.iloc[:, data.columns == 'quality'].values.ravel()

scaler = MinMaxScaler()
scaler.fit(X)
X_transf = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_transf, y, test_size=0.2, random_state=42)


classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)
y_results = classifier.predict(X_test)

# show_confusion_matrix(classifier, X_train, y_train)

precision = precision_score(y_test, y_results, average="micro") # Precision = TP / (TP + FP)
recall = recall_score(y_test, y_results, average="micro") # Recall = TP / (TP + FN)
print(precision, recall)