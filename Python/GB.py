# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 09:26:32 2016

@author: Sayam Ganguly
"""

#!/usr/bin/python
# -*- coding: utf8 -*-

# SAMPLE SUBMISSION TO THE BIG DATA HACKATHON 13-14 April 2013 'Influencers in a Social Network'
# .... more info on Kaggle and links to go here
#
# written by Ferenc Huszár, PeerIndex

from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPClassifier
import numpy as np
import os as os
from sklearn import cross_validation
from sklearn import ensemble

###########################
# LOADING TRAINING DATA
###########################
os.chdir('C:/Users/Sayam Ganguly/.spyder-py3/Influencer')
trainfile = open('train.csv')
for line in trainfile:
    header = line.rstrip().split(',')
    break

y_train = []
X_train_A = []
X_train_B = []
X_log = []

for line in trainfile:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    A_features = [float(item) for item in splitted[1:]]
    A_log = [float(item) for item in splitted[1:12]]
    B_features = [float(item) for item in splitted[12:]]
    y_train.append(label)
    X_train_A.append(A_features)
    X_log.append(A_log)
    X_train_B.append(B_features)
trainfile.close()

y_train = np.array(y_train)
X_train_A = np.array(X_train_A)
X_log = np.array(X_log)
X_train_B = np.array(X_train_B)
def slice_features(X):
    return X[:,[0,2,3,4,6]]
#
#X_train_A = slice_features(X_train_A)
#X_train_B = slice_features(X_train_B)
#print ("Mean" ,np.mean(X_train_A,axis=0))
#print("Shape", np.mean(X_train_A,axis=0).shape)

###########################
# EXAMPLE BASELINE SOLUTION USING SCIKIT-LEARN
#
# using scikit-learn LogisticRegression module without fitting intercept
# to make it more interesting instead of using the raw features we transform them logarithmically
# the input to the classifier will be the difference between transformed features of A and B
# the method roughly follows this procedure, except that we already start with pairwise data
# http://fseoane.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
###########################

def transform_features(x):
    return np.log(1+x)
#    x = (x - np.mean(x,axis = 0))/np.std(x,axis = 0)
#    return x

X_train = transform_features(X_train_A)#
#X_train_log = transform_features(X_log) - transform_features(X_train_B)
#print(X_train)
#np.savetxt("mean.csv",X_train,delimiter=',')
#model_logistic = linear_model.LogisticRegression()
#class BaseClassifier(ensemble.RandomForestClassifier):
#    def predict(self,X):
#        return self.predict_proba(X)
#    def fit(self,X,y,w):
#        return self.fit(X,y,w)
#
#model_logistic = BaseClassifier()
model = ensemble.GradientBoostingClassifier()

#model = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='logistic',
#                      hidden_layer_sizes=(5,2), random_state=1)

print("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model, X_train, y_train, cv=20, scoring='roc_auc')))
#model=SVC()
model.fit(X_train,y_train)
#model_logistic.fit(X_train_log,y_train)
# compute AuC score on the training data (BTW this is kind of useless due to overfitting, but hey, this is only an example solution)
p_train = model.predict_proba(X_train)
p_train = p_train[:,1:2]
precision, recall, thresholds=precision_recall_curve(y_train,p_train[:,0])
print('AuC score on training data:',roc_auc_score(y_train,p_train[:,0]))

###########################
# READING TEST DATA
###########################

testfile = open('test.csv')
#ignore the test header
for line in testfile:
    break

X_test_A = []
X_test_B = []
X_test_log = []
for line in testfile:
    splitted = line.rstrip().split(',')
    A_features = [float(item) for item in splitted[0:]]
    A_features_log = [float(item) for item in splitted[0:11]]
    B_features = [float(item) for item in splitted[11:]]
    X_test_A.append(A_features)
    X_test_log.append(A_features_log)
    X_test_B.append(B_features)
testfile.close()

X_test_A = np.array(X_test_A)
X_test_B = np.array(X_test_B)
#X_test_log = np.array(X_test_log)
#X_test_A = slice_features(X_test_A)
#X_test_B = slice_features(X_test_B)

# transform features in the same way as for training to ensure consistency
X_test = transform_features(X_test_A)
#X_test_log = transform_features(X_test_log) - transform_features(X_test_B)
# compute probabilistic predictions
p_test = model.predict_proba(X_test)
#p_test_logistic = model_logistic.predict_proba(X_test_log)
#only need the probability of the 1 class
p_test = p_test[:,1:2]
#p_test_logistic = p_test_logistic[:,1:2]

#p_test = p_test[:,0]
#p_test[:,-1] = (p_test[:,0] + p_test_logistic[:,0])/2
###########################
# WRITING SUBMISSION FILE
###########################
predfile = open('predictions.csv','w+')
predfile.write('Id,Choice,\n')
i=1;
for line in p_test:
    x=str(i)+','+str(line[0])
    predfile.write(x)
    #predfile.write(','.join(str(i)).','.join([str(item) for item in line]))
    predfile.write(',\n')
    i=i+1

predfile.close()