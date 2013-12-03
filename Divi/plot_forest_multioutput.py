# import numpy as np

# from sklearn.datasets import make_classification
# from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
# X, y = make_classification(n_samples=1000,
#                            n_features=10,
#                            n_informative=3,
#                            n_redundant=0,
#                            n_repeated=0,
#                            n_classes=2,
#                            random_state=0,
#                            shuffle=False)

# Build a forest and compute the feature importances
# forest = ExtraTreesClassifier(n_estimators=250,
#                               compute_importances=True,
#                               random_state=0)

# forest.fit(X, y)
# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]

# Print the feature ranking
# print "Feature ranking:"

# for f in xrange(10):
# print "%d. feature %d (%f)" % (f + 1, indices[f],
# importances[indices[f]])

# Plot the feature importances of the trees and of the forest
# import pylab as pl
# pl.figure()
# pl.title("Feature importances")

# for tree in forest.estimators_:
#     pl.plot(xrange(10), tree.feature_importances_[indices], "r")

# pl.plot(xrange(10), importances[indices], "b")
# pl.show()


import numpy as np
import pandas as pd
import random

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

target_names = ['New York', 'London', 'DC']

X_train = np.array(["new york is a hell of a town",
                    "new york was originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york",
                    "DC is the nations capital",
                    "DC the home of the beltway",
                    "president obama lives in Washington",
                    "The washington monument in is Washington DC"])

y_train = [[0], [0], [0], [0], [0], [0], [1], [1], [1],
          [1], [1], [1], [1, 0], [1, 0], [2], [2], [2], [2]]


X_test = np.array(['nice day in nyc',
                   'welcome to london',
                   'hello welcome to new ybrk. enjoy it here and london too',
                   'What city does the washington redskins live in?'])
y_test = [[0], [1], [0, 1], [2]]

classifier = Pipeline([
                      ('vectorizer', CountVectorizer(stop_words='english',
                                                     ngram_range=(1, 3),
                                                     max_df=1.0,
                                                     min_df=0.1,
                                                     analyzer='word')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, y_train)

predicted = classifier.predict(X_test)

print predicted


for item, labels in zip(X_test, predicted):
    print '%s => %s' % (item, ', '.join(target_names[x] for x in labels))


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss


hl = hamming_loss(y_test, predicted, target_names)
print " "
print " "
print "---------------------------------------------------------"
print "HAMMING LOSS"
print " "
print hl

print " "
print " "
print "---------------------------------------------------------"
print "CONFUSION MATRIX"
print " "
# cm = confusion_matrix(y_test, predicted)
# print cm

print " "
print " "
print "---------------------------------------------------------"
print "CLASSIFICATION REPORT"
print " "
print classification_report(y_test, predicted)
