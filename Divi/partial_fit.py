from __future__ import division
import csv
import nltk
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier, LabelBinarizer
from nltk.corpus import stopwords
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.pls import CCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import (LinearRegression, Lasso, ElasticNet, Ridge,
                                  Perceptron)
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm
from sklearn import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

id_list = []
title_list = []
question_list = []
keyword_list = []
test_id_list = []
test_title_list = []
test_question_list = []
test_keyword_list = []
threshold = .8
train_file_path = 'C:\\NLP_Data\\Train\\Train.csv'
test_file_path = 'C:\\NLP_Data\\Train\\Train.csv'


# def data_cleaner(raw_text):
#     text = re.sub("<p>|</p>|<pre>|<code>|</pre>|</code>", '', raw_text)
#     text = re.sub("\n|[,!;?:/']", ' ', text)
#     text = text.split(" ")
#     li = []
#     for w in text:
#         w = w.lower()
# w = [x.lower()
# for x in w if not x in stopwords.words('english')]
#         if w != "":
#             if w[-1] == ".":
#                 w = w[:-1]
#             li.append(w)
#     return str(li)


# def code_extractor(raw_text):
#     m = re.findall('<pre><code>(.+?)</code></pre>', raw_text, re.S)
#     return str(m)

num_of_questions = 10

# classifier = Pipeline([
#     ('vectorizer', CountVectorizer(min_df=1)),
#     ('tfidf', TfidfTransformer()),
#     ('clf', OneVsRestClassifier(LinearSVC()))])

# classifier = Pipeline([
#     ('vectorizer', CountVectorizer(min_df=1, max_df=0.8)),
#     ('tfidf', TfidfTransformer(use_idf=True)),
#     ('clf', OneVsRestClassifier(PassiveAggressiveClassifier(C=1, n_iter=1, n_jobs=2)))])

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8, min_df=1,
                             stop_words='english')

classifier = PassiveAggressiveClassifier(C=1, n_iter=1, n_jobs=1)

print "File reading started"
with open(train_file_path, 'r') as f:
    reader = csv.reader(f)
    for i in range(num_of_questions):
        line = reader.next()
        id_list.append(line[0])
        # inputtext = code_extractor(line[2])  # + line[2]
        inputtext = line[1]
        title_list.append(inputtext)
        # question_list.append(line[2])
        keyword_list.append(line[3])
print "File reading completed"

y = []
for i in keyword_list:
    y.append([x for x in str(i).split()])
# print y

X_train = np.array(title_list[:int(threshold * len(title_list))])
y_train = y[:int(threshold * len(title_list))]

X_test = np.array(title_list[int(threshold * len(title_list)):])
target_names = y[int(threshold * len(title_list)):]

print "Data loaded in lists!"

# print "X_train\n", X_train[:10]
# print "X_test\n", X_test[:10]
# print "y_train\n", y_train[:10]
# print "y_test\n", y_test

print "LabelBinarizer working!"

lb = LabelBinarizer()
Y = lb.fit_transform(y_train)
print "Fitting completed!"

print "Building classifier!"
# classifier.fit(X_train, Y)
classifier.partial_fit(X_train[:5], Y[:5])
classifier.partial_fit(X_train[5:], Y[5:])


print "Predicting!"
predicted = classifier.predict(X_test)

print "inverse_transform!"
all_labels = lb.inverse_transform(predicted)
print "Lengths", len(X_train), len(y_train), len(X_test), len(target_names)
print ""
print "Length = ", len(set(all_labels)) - 1
print "Percentage Predicted = ", (len(set(all_labels)) - 1) / len(target_names) * 100

# print "Printing stuff!"
# for item, labels in zip(X_test, all_labels):
#     if labels != ():
# print '%s => %s' % (item, ', '.join(labels))
# print '%s' % (', '.join(labels))
#         pass
# print classifier.get_params(deep=True)
