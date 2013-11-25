import csv
c = csv.writer(open("train_1000.csv", "wb"))
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    word_assoc_list = []
    word_list = []
    num_of_questions = 100
    for i in range(num_of_questions + 1):
        row = reader.next()
        c.writerow(row)
        if row[3] != "Tags":
            print "keywords[" + str(i) + "] " + row[3]
            for word in row[3].split(" "):
                if not word in word_list:
                    word_assoc_list.append((word, [associated_word for associated_word in row[3].split(" ")]))
                    word_list.append(word)
                else:
                    entry_id = word_list.index(word)
                    for word in row[3].split(" "):
                        if word not in word_assoc_list[entry_id][1]:
                            word_assoc_list[entry_id][1].append(word)
    print "\n*** word_assoc_list (num_of_questions = " + str(num_of_questions) + ") ***\n"
    for entry in sorted(word_assoc_list):
        print entry             

# keywords_100 = ['c#', 'java', 'php', 'javascript', 'android', 'jquery', 'c++', 'python', 'iphone', 'asp.net', 'mysql', 'html', '.net', 'ios', 'objective-c', 'sql', 'css', 'linux', 'ruby-on-rails', 'windows', 'c', 'sql-server', 'ruby', 'wpf', 'xml', 'ajax', 'database', 'regex', 'windows-7', 'asp.net-mvc', 'xcode', 'django', 'osx', 'arrays', 'vb.net', 'eclipse', 'json', 'facebook', 'ruby-on-rails-3', 'ubuntu', 'performance', 'networking', 'string', 'multithreading', 'winforms', 'security', 'visual-studio-2010', 'asp.net-mvc-3', 'bash', 'homework', 'image', 'wcf', 'html5', 'wordpress', 'visual-studio', 'web-services', 'forms', 'algorithm', 'sql-server-2008', 'linq', 'oracle', 'git', 'query', 'perl', 'apache2', 'flash', 'actionscript-3', 'ipad', 'spring', 'apache', 'silverlight', 'email', 'r', 'cocoa-touch', 'cocoa', 'swing', 'hibernate', 'excel', 'entity-framework', 'file', 'shell', 'flex', 'api', 'list', 'internet-explorer', 'firefox', 'jquery-ui', 'delphi', '.htaccess', 'sqlite', 'qt', 'tsql', 'google-chrome', 'node.js', 'unix', 'windows-xp', 'http', 'svn', 'unit-testing', 'oop']
