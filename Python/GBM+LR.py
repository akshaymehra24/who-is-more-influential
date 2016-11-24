# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:49:37 2016

@author: Akshay
"""

from sklearn import ensemble
from sklearn import linear_model
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import cross_validation
import os as os

###########################
# LOADING TRAINING DATA
###########################
os.chdir('C:/Users/aksha/OneDrive/Documents/GitHub/who-is-more-influential/Original Dataset')
trainfile = open('train.csv')
for line in trainfile:
    header = line.rstrip().split(',')
    break

y_train = []
X_train_A = []
X_train_B = []

for line in trainfile:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    A_features = [float(item) for item in splitted[1:12]]
    B_features = [float(item) for item in splitted[12:]]
    y_train.append(label)
    X_train_A.append(A_features)
    X_train_B.append(B_features)
trainfile.close()

y_train = np.array(y_train)
X_train_A = np.array(X_train_A)
X_train_B = np.array(X_train_B)

def transform_features(x):
    return np.log(1+x)

X_train = transform_features(X_train_A) - transform_features(X_train_B)
model1=ensemble.GradientBoostingClassifier().fit(X_train, y_train)
model2=linear_model.LogisticRegression().fit(X_train, y_train)

#20 Fold Cross Validation Score
print("20 Fold CV Score with GBM: ", np.mean(cross_validation.cross_val_score(model1, X_train, y_train, cv=20, scoring='roc_auc')))
print("20 Fold CV Score with GBM: ", np.mean(cross_validation.cross_val_score(model1, X_train, y_train, cv=20, scoring='roc_auc')))



