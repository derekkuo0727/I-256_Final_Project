import csv
import nltk


# def getScore(entry):
keyword_list = []

# c = csv.writer(open("C:\\NLP_Data\\Train\\train_1000.csv", "w"))
with open('C:\NLP_Data\Train\Train.csv', 'rb') as f:
    reader = csv.reader(f)
    word_assoc_list = []
    word_list = []
    num_of_questions = 3000
    for i in range(num_of_questions + 1):
        row = reader.next()
        # c.writerow(row)
        if row[3] != "Tags":
            keyword_list.append(row[3])
            # print "keywords[" + str(i) + "] " + row[3]
            for word in row[3].split(" "):
                if not word in word_list:
                    word_assoc_list.append(
                        (word, [associated_word for associated_word in row[3].split(" ")]))
                    word_list.append(word)
                else:
                    entry_id = word_list.index(word)
                    for word in row[3].split(" "):
                        if word not in word_assoc_list[entry_id][1]:
                            word_assoc_list[entry_id][1].append(word)
    print "\n*** word_assoc_list (num_of_questions = " + str(num_of_questions) + ") ***\n"
    for entry in sorted(word_assoc_list):
        print entry

fq = nltk.FreqDist(
    [keyword for keywords in keyword_list for keyword in keywords.split(" ")])
fq.plot(10)
