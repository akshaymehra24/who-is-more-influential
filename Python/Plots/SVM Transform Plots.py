# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 00:05:14 2016

@author: Akshay
"""


from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt 
import os as os

###########################
# LOADING TRAINING DATA
###########################
os.chdir('C:/Users/aksha/OneDrive/Documents/GitHub/who-is-more-influential/Data Analysis')
trainfile = open('train_new_discrete.csv')
for line in trainfile:
    header = line.rstrip().split(',')
    break

y_train = []
X_train_A = []

for line in trainfile:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    A_features = [float(item) for item in splitted[1:]]
    y_train.append(label)
    X_train_A.append(A_features)
trainfile.close()

y_train = np.array(y_train)
X_train_A = np.array(X_train_A)

def transform_features(x):
    return x

X_train = transform_features(X_train_A)
X_train = X_train[:,[0,1]]

C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C)
print("Model Trained")
svc.fit(X_train, y_train)
print("Model fitted")
#rbf_svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
#poly_svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
#lin_svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)

h = .02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# title for the plots
titles = ['Followers Vs Following']


for i, clf in enumerate((svc, svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(1, 1, i + 1)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    #plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
    plt.xlabel('Followers')
    plt.ylabel('Following')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()

