import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier, LabelBinarizer
import csv
import nltk
import logging

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
num_of_questions = 20000

print "File reading started"
with open(train_file_path, 'r') as f:
    reader = csv.reader(f)
    for i in range(num_of_questions + 1):
        line = reader.next()
        id_list.append(line[0])
        title_list.append(line[1])
        question_list.append(line[2])
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

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

print "Building classifier!"
classifier.fit(X_train, Y)

print "Predicting!"
predicted = classifier.predict(X_test)

print "inverse_transform!"
all_labels = lb.inverse_transform(predicted)
print len(X_train), len(y_train), len(X_test), len(target_names)

print "Printing stuff!"
for item, labels in zip(X_test, all_labels):
    print '%s => %s' % (item, ', '.join(labels))
