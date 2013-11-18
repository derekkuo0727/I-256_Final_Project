import csv
from sets import Set

cr = csv.reader(open("train_1000.csv","rb"))
match_set = Set()

# Ask for the keyword and store it in userKeyword
userKeyword = raw_input('Give me a keyword: ')

# if match, store in this format [row1, row2, row3...]
title_found = []
body_found = []
tag_found = []
for row in cr:
    if userKeyword.lower() in row[1].lower().split(" "): # otherwise "java" matches "javascript"
        title_found.append(row)
        match_set.add(row[0])
    if userKeyword.lower() in row[2].lower().split(" "):
        body_found.append(row)
        match_set.add(row[0])
    if userKeyword.lower() in row[3].lower().split(" "):
        tag_found.append(row)
        match_set.add(row[0])

# Print search result of keyword in train_100.csv
print "============================"
print "===========summary=========="
print "============================"
print ""

print "##### title_found #####"
for entry in title_found:
    print "[ID: " + entry[0] + "]"
    print entry[1]
print ""

print "##### body_found #####"
for entry in title_found:
    print "[ID: " + entry[0] + "]"
    print entry[2]
print ""

print "##### tag_found #####"
for entry in title_found:
    print "[ID: " + entry[0] + "]"
    print entry[3]
print ""

print "##### overall match result (ID) #####"
print match_set
print "total number of match: " + str(len(match_set))





