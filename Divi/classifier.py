

import csv
import nltk
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

num_of_questions = 30
test_num_of_questions = 60


def benchmark(clf):
    print 80 * '_'
    print "Training: "
    print clf
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print "test time:  %0.3fs" % test_time

    score = metrics.f1_score(y_test, pred)
    print "f1-score:   %0.3f" % score

    if hasattr(clf, 'coef_'):
        print "dimensionality: %d" % clf.coef_.shape[1]
        print "density: %f" % density(clf.coef_)

    print
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

id_list = []
title_list = []
question_list = []
keyword_list = []

test_id_list = []
test_title_list = []
test_question_list = []
test_keyword_list = []

with open('C:\NLP_Data\Train\Train.csv', 'rb') as f:
    reader = csv.reader(f)
    for i in range(num_of_questions + 1):
        line = reader.next()
        id_list.append(line[0])
        title_list.append(line[1])
        question_list.append(line[2])
        keyword_list.append(line[3])

with open('C:\NLP_Data\Train\Train.csv', 'rb') as test_f:
    test_reader = csv.reader(test_f)
    for i in range(test_num_of_questions, test_num_of_questions + 10):
        test_line = test_reader.next()
        test_id_list.append(test_line[0])
        test_title_list.append(test_line[1])
        test_question_list.append(test_line[2])
        test_keyword_list.append(test_line[3])


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
print vectorizer
X_train = vectorizer.fit_transform(title_list)
X_test = vectorizer.transform(test_title_list)
y_train, y_test = keyword_list, test_keyword_list

print "X_train\n", X_train
print "X_test\n", X_test
print "y_train\n", y_train
print "y_test\n", y_test

results = []
print "Naive Bayes"
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

# for penalty in ["l2", "l1"]:
#     print 80 * '='
#     print "%s penalty" % penalty.upper()
# Train Liblinear model
#     results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
#                                             dual=False, tol=1e-3)))

# Train SGD model
#     results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                            penalty=penalty)))

# Train SGD with Elastic Net penalty
# print 80 * '='
# print "Elastic-Net penalty"
# results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                        penalty="elasticnet")))

# Train NearestCentroid without threshold
# print 80 * '='
# print "NearestCentroid (aka Rocchio classifier)"
# results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
# print 80 * '='
# print "Naive Bayes"
# results.append(benchmark(MultinomialNB(alpha=.01)))
# results.append(benchmark(BernoulliNB(alpha=.01)))


# class L1LinearSVC(LinearSVC):

#     def fit(self, X, y):
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
#         self.transformer_ = LinearSVC(penalty="l1",
#                                       dual=False, tol=1e-3)
#         X = self.transformer_.fit_transform(X, y)
#         return LinearSVC.fit(self, X, y)

#     def predict(self, X):
#         X = self.transformer_.transform(X)
#         return LinearSVC.predict(self, X)

# print 80 * '='
# print "LinearSVC with L1-based feature selection"
# results.append(benchmark(L1LinearSVC()))


# make some plots

# indices = np.arange(len(results))

# results = [[x[i] for x in results] for i in xrange(4)]

# clf_names, score, training_time, test_time = results

# pl.title("Score")
# pl.barh(indices, score, .2, label="score", color='r')
# pl.barh(indices + .3, training_time, .2, label="training time", color='g')
# pl.barh(indices + .6, test_time, .2, label="test time", color='b')
# pl.yticks(())
# pl.legend(loc='best')
# pl.subplots_adjust(left=.25)

# for i, c in zip(indices, clf_names):
#     pl.text(-.3, i, c)

# pl.show()
