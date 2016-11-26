# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:49:37 2016

@author: Akshay
"""

from sklearn import ensemble
from sklearn import linear_model
import numpy as np
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
model1=ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=0).fit(X_train, y_train)
model2=linear_model.LogisticRegression(fit_intercept=False).fit(X_train, y_train)
model3=ensemble.RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0).fit(X_train,y_train)

#20 Fold Cross Validation Score
print("20 Fold CV Score with GBM: ", np.mean(cross_validation.cross_val_score(model1, X_train, y_train, cv=20, scoring='roc_auc')))
print("20 Fold CV Score with Logistic: ", np.mean(cross_validation.cross_val_score(model2, X_train, y_train, cv=20, scoring='roc_auc')))
print("20 Fold CV Score with Random Forest: ", np.mean(cross_validation.cross_val_score(model3, X_train, y_train, cv=20, scoring='roc_auc')))
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
p_test1 = model1.predict_proba(X_test)
p_test2 = model2.predict_proba(X_test)
p_test3 = model3.predict_proba(X_test)
#only need the probability of the 1 class
p_test1[:,0] = p_test1[:,0]*0.0 + p_test2[:,0]*1 + p_test3[:,0]*0.0


###########################
# WRITING SUBMISSION FILE
###########################
predfile = open('predictions.csv','w+')
predfile.write('Id,Choice\n')
i=1;
for line in p_test1:
    x=str(i)+','+str(line[0])
    predfile.write(x)
    predfile.write('\n')
    i=i+1
predfile.close()

