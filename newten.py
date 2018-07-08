# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 01:07:44 2018

@author: roiky
"""

from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier

dataset = pd.read_csv('tennis_dataset.csv')

print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

X = dataset.iloc[:, 2:-1].values
Y = dataset.iloc[:,10].values

print(dataset.info())
print(dataset.describe())

clf1 = RandomForestClassifier()
clf2 = SGDClassifier()
clf3 = svm.SVC(kernel='linear',C=0.4)
clf4 = GaussianNB()
clf5 = ExtraTreesClassifier(random_state=0)

accuracy_clf1 = []
accuracy_clf2 = []
accuracy_clf3 = []
accuracy_clf4 = []
accuracy_clf5 = []

precision_clf1= []
precision_clf2 = []
precision_clf3 = []
precision_clf4= []
precision_clf5= []

recall_clf1 = []
recall_clf2= []
recall_clf3 = []
recall_clf4 = []
recall_clf5 = []

F1_clf1 = []
F1_clf2 = []
F1_clf3 = []
F1_clf4 = []
F1_clf5 = []

kf = KFold(n_splits=5,random_state=33,shuffle = True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    scaler = preprocessing.StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf1.fit(X_train, Y_train)
    clf2.fit(X_train, Y_train)
    clf3.fit(X_train, Y_train)
    clf4.fit(X_train, Y_train)
    clf5.fit(X_train, Y_train)
    
    y_pred_clf1= clf1.predict(X_test)
    y_pred_clf2 = clf2.predict(X_test)
    y_pred_clf3 = clf3.predict(X_test)
    y_pred_clf4 = clf4.predict(X_test)
    y_pred_clf5 = clf5.predict(X_test)
    
    accuracy_clf1.append( metrics.accuracy_score(Y_test, y_pred_clf1))
    accuracy_clf2.append( metrics.accuracy_score(Y_test, y_pred_clf2))
    accuracy_clf3.append( metrics.accuracy_score(Y_test, y_pred_clf3))
    accuracy_clf4.append( metrics.accuracy_score(Y_test, y_pred_clf4))
    accuracy_clf5.append( metrics.accuracy_score(Y_test, y_pred_clf5))
    
    precision_clf1.append(metrics.precision_score(Y_test, y_pred_clf1,average='macro'))
    precision_clf2.append(metrics.precision_score(Y_test, y_pred_clf2,average='macro'))
    precision_clf3.append(metrics.precision_score(Y_test, y_pred_clf3,average='macro'))
    precision_clf4.append(metrics.precision_score(Y_test, y_pred_clf4,average='macro'))
    precision_clf5.append(metrics.precision_score(Y_test, y_pred_clf5,average='macro'))

    recall_clf1.append(metrics.recall_score(Y_test, y_pred_clf1,average='macro'))
    recall_clf2.append(metrics.recall_score(Y_test, y_pred_clf2,average='macro'))
    recall_clf3.append(metrics.recall_score(Y_test, y_pred_clf3,average='macro'))
    recall_clf4.append(metrics.recall_score(Y_test, y_pred_clf4,average='macro'))
    recall_clf5.append(metrics.recall_score(Y_test, y_pred_clf5,average='macro'))
    
    
    F1_clf1.append(metrics.f1_score(Y_test, y_pred_clf1,average='macro'))
    F1_clf2.append(metrics.f1_score(Y_test, y_pred_clf2,average='macro'))
    F1_clf3.append(metrics.f1_score(Y_test, y_pred_clf3,average='macro'))
    F1_clf4.append(metrics.f1_score(Y_test, y_pred_clf4,average='macro'))
    F1_clf5.append(metrics.f1_score(Y_test, y_pred_clf5,average='macro'))
    
print ("Random Forest Classifier:")
print ("Accuracy : ",np.mean(accuracy_clf1))
print ("Precision : ",np.mean(precision_clf1))
print ("Recall : ",np.mean(recall_clf1))
print ("F1 : ",np.mean(F1_clf1))
print ("       ")

print ("SGD Classifier:")
print ("Accuracy : ",np.mean(accuracy_clf2))
print ("Precision : ",np.mean(precision_clf2))
print ("Recall : ",np.mean(recall_clf2))
print ("F1 : ",np.mean(F1_clf2))
print ("       ")


print ("Support Vector Classification:")
print ("Accuracy : ",np.mean(accuracy_clf3))
print ("Precision : ",np.mean(precision_clf3))
print ("Recall : ",np.mean(recall_clf3))
print ("F1 : ",np.mean(F1_clf3))
print ("       ")


print ("GaussianNB Classifier:")
print ("Accuracy : ",np.mean(accuracy_clf4))
print ("Precision : ",np.mean(precision_clf4))
print ("Recall : ",np.mean(recall_clf4))
print ("F1 : ",np.mean(F1_clf4))
print ("       ")

print ("ExtraTrees Classifier:")
print ("Accuracy : ",np.mean(accuracy_clf5))
print ("Precision : ",np.mean(precision_clf5))
print ("Recall : ",np.mean(recall_clf5))
print ("F1 : ",np.mean(F1_clf5))
print ("       ")
