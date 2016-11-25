# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 19:47:40 2016

@author: Akshay
"""

from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
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

def transform_features_log(x):
    return np.log(1+x)  

X_train_log = transform_features_log(X_train_A) - transform_features_log(X_train_B)
X_train_log = X_train_log[:,[0,2]]

lm_log = linear_model.LogisticRegression().fit(X_train_log, y_train)

h = .02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = X_train_log[:, 0].min() - 1, X_train_log[:, 0].max() + 1
y_min, y_max = X_train_log[:, 1].min() - 1, X_train_log[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# title for the plots
titles = ['Following Vs NF3']


for i, clf in enumerate((lm_log, lm_log)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(1, 1, i + 1)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X_train_log[:, 0], X_train_log[:, 1], c=y_train, cmap=plt.cm.coolwarm)
    plt.xlabel('Following')
    plt.ylabel('NF3')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
