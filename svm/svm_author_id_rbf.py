#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf = svm.SVC(C=10000.0,kernel='rbf',gamma='auto')

t0 = time()
clf.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print ("predicting time:", round(time()-t1, 3), "s")

accuracy = accuracy_score(pred,labels_test)
print(accuracy)

#predict for element 10 of the test set, The 26th, The 50th
print ("predicting 10:", pred[10])
print ("predicting 26:", pred[26])
print ("predicting 50:", pred[50])

#predict for Chris's 1 class
print ("predicting for Chris's 1 class:", sum(pred))
#########################################################
### your code goes here ###

#########################################################


