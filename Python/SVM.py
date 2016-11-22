# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:28:06 2016

@author: Akshay
"""

from sklearn import svm
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
model=svm.SVC(probability=True);
C = 1.0  # SVM regularization parameter
model = svm.SVC(kernel='linear', C=C, probability=True).fit(X_train, y_train) #SVM with Linear Kernel
#model = svm.SVC(kernel='rbf', gamma=0.7, C=C, probability=True).fit(X_train, y_train) #rbf_svm
#model = svm.SVC(kernel='poly', degree=3, C=C, probability=True).fit(X_train, y_train) #Polynomial SVM
#model = svm.LinearSVC(C=C).fit(X_train, y_train) #Linear SVM

#20 Fold Cross Validation Score
print("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model, X_train, y_train, cv=20, scoring='roc_auc')))


# compute AuC score on the training data (BTW this is kind of useless due to overfitting, but hey, this is only an example solution)
p_train = model.predict_proba(X_train)
p_train = p_train[:,1:2]
precision, recall, thresholds=precision_recall_curve(y_train,p_train[:,0])
print('AuC score on training data:',roc_auc_score(y_train,p_train[:,0]))

###########################
# READING TEST DATA
###########################

testfile = open('C:\\Users\\aksha\\OneDrive\\Documents\\GitHub\\who-is-more-influential\\Original Dataset\\test.csv')
#ignore the test header
for line in testfile:
    break

X_test_A = []
X_test_B = []
for line in testfile:
    splitted = line.rstrip().split(',')
    A_features = [float(item) for item in splitted[0:11]]
    B_features = [float(item) for item in splitted[11:]]
    X_test_A.append(A_features)
    X_test_B.append(B_features)
testfile.close()

X_test_A = np.array(X_test_A)
X_test_B = np.array(X_test_B)

# transform features in the same way as for training to ensure consistency
X_test = transform_features(X_test_A) - transform_features(X_test_B)

# compute probabilistic predictions
p_test = model.predict_proba(X_test)
#only need the probability of the 1 class
p_test = p_test[:,1:2]

###########################
# WRITING SUBMISSION FILE
###########################
predfile = open('predictions.csv','w+')
predfile.write('Id,Choice\n')
i=1;
for line in p_test:
    x=str(i)+','+str(line[0])
    predfile.write(x)
    predfile.write('\n')
    i=i+1

predfile.close()