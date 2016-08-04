# -*- coding: utf-8 -*-
"""First data science steps learning from
https://github.com/mbesson/TripAdvisor-datamining/blob/master/rapport/Analyse-des-Sentiments.ipynb
and using scikitlearn"""

import pandas as pd
import time
from sklearn import svm

data_set = pd.read_csv("train.csv")
keys = data_set.keys()
category_size = 50
SVM_keys = keys[1:6]

dates = data_set['datetime']
counts = data_set['count']

"""Using a SVM to classify the training data
into bike rental counts categories.
category 1: 0-category_size bikes
etc...
"""

print "SVM keys, ", SVM_keys
print "category size, ", category_size 

features = []

for i in range(8000):
    features.append(
        [data_set[key][i] for key in SVM_keys])

categories = [count / category_size for count in counts[0:8000]]

clf = svm.SVC()
t1 = time.clock()
clf.fit(features, categories)
t2 = time.clock()

print "fit done in %.3f seconds"%(t2-t1)

test_features = []

for i in range(8000, len(data_set)):
    test_features.append(
        [data_set[key][i] for key in SVM_keys])

test_categories = [count / category_size for count in counts[8000:]]

t1 = time.clock()
predicted_categories = clf.predict(test_features)
t2 = time.clock()

print "prediction done in %.3f seconds"%(t2-t1)

compteur = 0

for i in range(len(test_categories)):
    if test_categories[i] == predicted_categories[i]:
        compteur += 1

percent = compteur/float(len(test_categories))

print "success rate %.3f"%(percent)
