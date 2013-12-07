from __future__ import division

# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <rawcell>

# *** Tool function: get_line_num_by_key(key) ***
# Input: a keyword
# Output: a list of line numbers that each of the line contains the keyword.

# <codecell>

import numpy


def get_line_num_by_key(key):
    line_num = []
    for i in range(len(keyword_list)):
        if key in keyword_list[i].split(" "):
            line_num.append(i)
    return line_num

# <rawcell>

# *** Tool function: data_cleaner(raw_text) ***
# Input: a raw text directly extracted from question_lists or title_lists
# Ouput: a list of words (ready to use) which formats have been removed.

# <codecell>

import re


def data_cleaner(raw_text):
    text = re.sub("<p>|</p>|<pre>|<code>|</pre>|</code>", '', raw_text)
    text = re.sub("\n|[,!;?:/']", ' ', text)
    text = text.split(" ")
    li = []
    for w in text:
        w = w.lower()
        if w != "":
            if w[-1] == ".":
                w = w[:-1]
            li.append(w)
    return li


# <rawcell>

# *** Tool function: code_extractor(raw_text) ***
# Input: a raw text directly extracted from question_lists or title_lists
# Output: a list of code sections

# <codecell>

def code_extractor(raw_text):
    m = re.findall('<pre><code>(.+?)</code></pre>', raw_text, re.S)
    return m

# <rawcell>

# *** 0. Create a small train set (small_train.csv) from train.csv ***

# <codecell>

import csv

input_file = open("C:\\NLP_Data\\Train\\Train.csv", 'r')
output_file = open("C:\\NLP_Data\\Train\\output_small_train.csv", "w")
reader = csv.reader(input_file)

i = 0

for line in reader:
    a = csv.writer(output_file, delimiter=',')
    a.writerow(line)
    i += 1
    if i > 1000005:
        break

# <rawcell>

# *** 0.1 Variable Definition ***

# <codecell>

TRAINING_LINE_NUMBER = 100000
TOP_KEYWORDS = 50
TOP_WORDS = 200

# <rawcell>

# *** 1. Load small_train.csv into title, question, keyword list ***
# variable: ___ of lines to load

# <codecell>

import csv
import nltk

skip_first_line = True

input_file = open("small_train.csv", 'r')
reader = csv.reader(input_file)
if skip_first_line:
    reader.next()

#id_list = []
title_list = []
question_list = []
keyword_list = []

i = 0

for line in reader:
    id_list.append(line[0])
    title_list.append(line[1])
    question_list.append(line[2])
    keyword_list.append(line[3])
    i += 1
    if i == TRAINING_LINE_NUMBER:
        break
# Now you can just call the function:


# <rawcell>
# *** 2. Build-up freqency table to find the most frequent keywords ***
# <codecell>
fq = nltk.FreqDist(
    [keyword for keywords in keyword_list for keyword in keywords.split(" ")])

# <rawcell>

# *** 2.1 Checking_point: print the top x keywords and their frquency ***

# <codecell>

for i in range(TOP_KEYWORDS):
    key = fq.keys()[i]
    print key, fq[key]

# <rawcell>

# *** 3. Build the freq_word[keyword] dictionary (unigram version) ***
# Variable: top __ keyword
# Variable: top __ words in a keyword
#
# keyword: one of the top x keywords
# freq_word[keyword]: return a dictionary of {word: [tf, idf]}
# ex. freq_word['java'] ==> {"data": [123,234],"class": [23,32],...}
# which means 'data' has appeared 123 times inside keyword and 234 times
# outside keyword

# <codecell>

# unigram
import re

import pickle
tagger = pickle.load(open("treebank_brill_aubt.pickle"))

from nltk.corpus import stopwords

freq_word = {}

# build tf
for keyword in fq.keys()[:TOP_KEYWORDS]:
    line_num = get_line_num_by_key(keyword)
    target_words = []
    for num in line_num:
        text = data_cleaner(question_list[num])
        text = tagger.tag(text)
        for (word, tag) in text:
            if (not word in stopwords.words('english'))and tag in ["JJ", "NN", "NNP", "NNS", "-None-"] and word not in ["", "=", "{", "}", "(", ")", "+", "=="]:
                    target_words.append(word.lower())
    fdist = nltk.FreqDist(target_words)
    key_value = {}
    for key in fdist.keys()[:TOP_WORDS]:
        key_value[key] = [fdist[key], 0]
    freq_word[keyword] = key_value

# build idf
for i in range(TRAINING_LINE_NUMBER):
    keywords = keyword_list[i].split(" ")
    text = data_cleaner(question_list[i])
    for keyword in freq_word.keys():
        if keyword not in keywords:
            for word in text:
                if word in freq_word[keyword].keys():
                    freq_word[keyword][word][1] += 1

# <codecell>

freq_word["c#"]

# <codecell>


# <rawcell>

# *** TODO: 3.2 Build the freq_word[keyword] dictionary (bigram version) ***

# <codecell>

# bigrams
"""
import re
import pickle
tagger = pickle.load(open("treebank_brill_aubt.pickle"))

from nltk import bigrams

freq_word = {}
for keyword in fq.keys()[:10]:
    line_num = get_line_num_by_key(keyword)
    fdist = nltk.FreqDist()
    for num in line_num:
        text = title_list[num]
        text = re.sub('[,.!;()?:/]', ' ', text)
        text = text.split(" ")
        text = tagger.tag(text)
        text = bigrams(text)

        for (word1,tag1),(word2,tag2) in text:
            if (tag1 in ["JJ","NN", "NNP", "NNS","-None-"] or tag2 in ["JJ","NN", "NNP", "NNS","-None-"]) and (word1 != "") and (word2 != ""):
                fdist.inc(word1.lower()+ " " + word2.lower())
    key_value = []
    for key in fdist.keys()[:200]:
        key_value.append((key,fdist[key]))
    freq_word[keyword] = key_value
"""

# <rawcell>

# *** 5. Ouput the TFIDF table to file ***
# Each line contains [keyword, word, tf/idf, tf, appear times in other
# keyword, idf, total frequency of keywords]

# <codecell>

f = open("tfidf", "w")

for keyword in fq.keys()[:TOP_KEYWORDS]:
    for word in freq_word[keyword].keys():
        tf = freq_word[keyword][word][0]
        idf = freq_word[keyword][word][1]
        f.write(str(keyword) + "\t" + str(word) + "\t" + str(tf / (idf + 1))
                + "\t" + str(tf) + "\t" + str(idf) + "\t" + str(fq[keyword]) + "\n")
f.close()

# <rawcell>

# *** 6.0 Evaluation Variable Definition **

# <codecell>

TFIDF_Threshold = 10
NUMBER_OF_MATCHES = 2
TESTING_LINES = 100, 200

# <rawcell>

# *** 6. Evaluation: Load the tfidf table to build up score table ***
# variable: load only the entries that td/idf score over __ score
# variable: specify which tfidf table want to load
#
# score_table is a word based dictionary that give a word as key return a list of keywords that has tfidf over __ score with the word
#
# ex. score_table["class"] ==> ["java","c#","c++",...]
#
# means the word "class" has strong relationship (tfidf > __) with
# following keywords: java, c#, c++

# <codecell>

score_file = open("tfidf", "r")

score_table = {}

for line in score_file:
    line = line.strip("\n")
    line = line.split("\t")
    if float(line[2]) > TFIDF_Threshold:
        if line[1] not in score_table:
            score_table[line[1]] = [line[0]]
        else:
            score_table[line[1]].append(line[0])

# <rawcell>

# *** 7. Evaluation: output the prediction keywords and compare with the correct keywords ***
# variable: # of lines to predict

# <codecell>

# Unigram
prediction_table = []
for i in range(len(title_list)):
    text = data_cleaner(question_list[i])
    key_word = {}
    prediction = []
    for word in text:
        if word in score_table:
            for j in range(len(score_table[word])):
                if score_table[word][j] not in key_word:
                    key_word[score_table[word][j]] = 1
                else:
                    key_word[score_table[word][j]] += 1
    for key in key_word.keys():
        if key_word[key] > NUMBER_OF_MATCHES:
            prediction.append(key)
    prediction_table.append(prediction)

for i in range(TESTING_LINES):
    print "line:", i
    print prediction_table[i]
    print keyword_list[i]

# <rawcell>

# *** 8. Evaluation tool: function accuracy_query(keyword) ***
# input: a keyword
# return: print the precision information about the keyword

# <codecell>


def accuracy_query(keyword):
    correct_guess = 0
    total_key = 0
    total_guess = 0
    for i in range(TESTING_LINES):
        for guess in prediction_table[i]:
            if guess == keyword:
                total_guess += 1
                if guess in keyword_list[i]:
                    correct_guess += 1
        for key in keyword_list[i].split(" "):
            if key == keyword:
                total_key += 1
    print "Total key is:", total_key
    print "Total guess is:", total_guess
    print "Correct guess is:", correct_guess
    print "Recall is:", correct_guess / total_key
    print "Accuracy is:", correct_guess / total_guess
    print "Recall is:", 2 * (correct_guess / total_key) * (correct_guess / total_guess) / (correct_guess / total_key + correct_guess / total_guess)


# <rawcell>
# *** 9. Test specific keyword accuracy ***
# <codecell>
accuracy_query("r")
