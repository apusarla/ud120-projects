#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
from pprint import pprint


sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments', 'total_stock_value'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data = featureFormat(data_dict, features_list)

outliers = []
salary = []
payplusstock = []


for point in data:
    poi = point[0]
    sal = point[1]
    salary.append(sal)
    pay = point[2] + point[3]
    payplusstock.append(pay)

    plt.scatter( sal, pay )
    outliers.append((int(sal),int(pay)))

import numpy
numpy.amax(data)

#pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[:2])


salary = numpy.reshape( numpy.array(salary), (len(salary), 1))
print("Salary", len(salary))
payplusstock = numpy.reshape( numpy.array(payplusstock), (len(payplusstock), 1))
print("payplusstock", len(payplusstock))

print(outliers)

from sklearn.model_selection import train_test_split
salary_train, salary_test, payplusstock_train, payplusstock_test = train_test_split(salary, payplusstock, test_size=0.5, random_state=42)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(salary_train, payplusstock_train)

try:
    plt.plot(salary, reg.predict(salary), color="blue")
except NameError:
    pass
plt.scatter(salary, payplusstock)
plt.show()




#print(data_dict.pop('TOTAL',0))

### Task 2: Remove outliers



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
print("Hai......")