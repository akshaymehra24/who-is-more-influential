# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:28:06 2016

@author: Akshay
"""

from sklearn import svm
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn import cross_validation
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
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
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.5, random_state=0)


model=svm.SVC(probability=True);
C = 1.0  # SVM regularization parameter
svc  = svm.SVC(kernel='linear', C=C, probability=True).fit(X_train, y_train) #SVM with Linear Kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C, probability=True).fit(X_train, y_train) #rbf_svm
poly_svc = svm.SVC(kernel='poly', degree=3, C=C, probability=True).fit(X_train, y_train) #Polynomial SVM
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train) #Linear SVM

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

plot_label = ['svc', 'rbf_svc', 'poly_svc', 'lin_svc']
plot_color = ['k', 'b-', 'r-', 'c-']

for i, clf in enumerate((svc, rbf_svc, poly_svc, svc)):
    y_pred=clf.predict_proba(X_test)[:, 1]
    # Analyze the results
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_label = '{0} {1:0.3f}'.format(plot_label[i], roc_auc)

    # Graph results
    if i==3:
        plt.plot(false_positive_rate, true_positive_rate, plot_color[i], linestyle='dashed' , label=roc_label, linewidth=2)
    elif i==0:
        plt.plot(false_positive_rate, true_positive_rate, plot_color[i], linestyle='dotted' ,label=roc_label, linewidth=2)
    else:
        plt.plot(false_positive_rate, true_positive_rate, plot_color[i], label=roc_label, linewidth=2)

# Graph Labels
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')     # plot the diagonal
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# plt.show()
plt.show()
