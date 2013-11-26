import itertools
import sys
import time


def tokenize(file_name):
    # Assumes that sentences are separated by a single '\n'.
    # Assumes that words are separated by a single ' '.
    # Tokenizes each sentence, removes duplicate tokens, sorts tokens.
    # print[sorted(list(set(e.split()))) for e in
    # open(file_name).read().strip().split('\n')]
    return [sorted(list(set(e.split()))) for e in
            open(file_name).read().strip().split('\n')]


def frequent_itemsets(sentences):
    # Counts sets with Apriori algorithm.
    SUPP_THRESHOLD = 3
    supps = []

    supp = {}
    for sentence in sentences:
        for key in sentence:
            if key in supp:
                supp[key] += 1
            else:
                supp[key] = 1
    print "|C1| = " + str(len(supp))
    supps.append({k: v for k, v in supp.iteritems() if v >= SUPP_THRESHOLD})
    print "|L1| = " + str(len(supps[0]))

    supp = {}
    for sentence in sentences:
        for combination in itertools.combinations(sentence, 2):
            if combination[0] in supps[0] and combination[1] in supps[0]:
                key = ','.join(combination)
                if key in supp:
                    supp[key] += 1
                else:
                    supp[key] = 1
    print "|C2| = " + str(len(supp))
    supps.append({k: v for k, v in supp.iteritems() if v >= SUPP_THRESHOLD})
    print "|L2| = " + str(len(supps[1]))

    supp = {}
    for sentence in sentences:
        for combination in itertools.combinations(sentence, 3):
            if (combination[0] + ',' + combination[1] in supps[1] and
                    combination[0] + ',' + combination[2] in supps[1] and
                    combination[1] + ',' + combination[2] in supps[1]):
                key = ','.join(combination)
                if key in supp:
                    supp[key] += 1
                else:
                    supp[key] = 1
    print "|C3| = " + str(len(supp))
    supps.append({k: v for k, v in supp.iteritems() if v >= SUPP_THRESHOLD})
    print "|L3| = " + str(len(supps[2]))

    return supps


def measures(supp_ab, supp_a, supp_b, transaction_count):
    # Assumes A -> B, where A and B are sets.
    conf = float(supp_ab) / float(supp_a)
    s = float(supp_b) / float(transaction_count)
    lift = conf / s
    if conf == 1.0:
        conv = float('inf')
    else:
        conv = (1 - s) / (1 - conf)
    return [conf, lift, conv]


def generate_rules(measure, supps, transaction_count):
    rules = []
    CONF_THRESHOLD = 0.85
    LIFT_THRESHOLD = 20.0
    CONV_THRESHOLD = 5.0
    if measure == 'conf':
        for i in range(2, len(supps) + 1):
            for k, v in supps[i - 1].iteritems():
                k = k.split(',')
                for j in range(1, len(k)):
                    for a in itertools.combinations(k, j):
                        b = tuple([w for w in k if w not in a])
                        [conf, lift, conv] = measures(v,
                                                      supps[len(a) - 1][
                                                      ','.join(a)],
                                                      supps[len(b) - 1][
                                                      ','.join(b)],
                                                      transaction_count)
                        if conf >= CONF_THRESHOLD:
                            rules.append((a, b, conf, lift, conv))
            rules = sorted(rules, key=lambda x: (x[0], x[1]))
            rules = sorted(rules, key=lambda x: (x[2]), reverse=True)
    elif measure == 'lift':
        for i in range(2, len(supps) + 1):
            for k, v in supps[i - 1].iteritems():
                k = k.split(',')
                for j in range(1, len(k)):
                    for a in itertools.combinations(k, j):
                        b = tuple([w for w in k if w not in a])
                        [conf, lift, conv] = measures(v,
                                                      supps[len(a) - 1][
                                                      ','.join(a)],
                                                      supps[len(b) - 1][
                                                      ','.join(b)],
                                                      transaction_count)
                        if lift >= LIFT_THRESHOLD:
                            rules.append((a, b, conf, lift, conv))
            rules = sorted(rules, key=lambda x: (x[0], x[1]))
            rules = sorted(rules, key=lambda x: (x[3]), reverse=True)
    elif measure == 'conv':
        for i in range(2, len(supps) + 1):
            for k, v in supps[i - 1].iteritems():
                k = k.split(',')
                for j in range(1, len(k)):
                    for a in itertools.combinations(k, j):
                        b = tuple([w for w in k if w not in a])
                        [conf, lift, conv] = measures(v,
                                                      supps[len(a) - 1][
                                                      ','.join(a)],
                                                      supps[len(b) - 1][
                                                      ','.join(b)],
                                                      transaction_count)
                        if conv >= CONV_THRESHOLD:
                            rules.append((a, b, conf, lift, conv))
            rules = sorted(rules, key=lambda x: (x[0], x[1]))
            rules = sorted(rules, key=lambda x: (x[4]), reverse=True)
    else:
        for i in range(2, len(supps) + 1):
            for k, v in supps[i - 1].iteritems():
                k = k.split(',')
                for j in range(1, len(k)):
                    for a in itertools.combinations(k, j):
                        b = tuple([w for w in k if w not in a])
                        [conf, lift, conv] = measures(v,
                                                      supps[len(a) - 1][
                                                      ','.join(a)],
                                                      supps[len(b) - 1][
                                                      ','.join(b)],
                                                      transaction_count)
                        if (conf >= CONF_THRESHOLD and
                                lift >= LIFT_THRESHOLD and
                                conv >= CONV_THRESHOLD):
                            rules.append((a, b, conf, lift, conv))
            rules = sorted(rules, key=lambda x: (x[0], x[1]))
            rules = sorted(
                rules, key=lambda x: (x[2], x[3], x[4]), reverse=True)
    return rules


def main():
    # if len(sys.argv) < 2:
    #     sys.stderr.write("Usage: python a_priori.py"
    #                      "transactions.txt [measure]\n")
    #     sys.exit(1)
    # elif len(sys.argv) == 2 or sys.argv[2] not in ['conf', 'lift', 'conv']:
    #     measure = 'all'
    # else:
    #     measure = sys.argv[2]

    # sentences = tokenize(sys.argv[1])
    sentences = tokenize('C:\Users\Divyakumar\Documents\GitHub\I-256_Final_Project\Divi\Assoc_Input.txt')
    measure = 'all'
    start_time = time.time()
    supps = frequent_itemsets(sentences)
    end_time = time.time()
    print "Time spent finding frequent itemsets = {:.2f} seconds.".format(
          end_time - start_time)

    start_time = time.time()
    rules = generate_rules(measure, supps, len(sentences))
    for rule in rules:
        print("{{{}}} -> {{{}}}, "
              "conf = {:.2f}, lift = {:.2f}, conv = {:.2f}").format(
                  ', '.join(rule[0]), ', '.join(rule[1]), rule[2], rule[3], rule[4])
    end_time = time.time()
    print "Time spent finding association rules = {:.2f} second.".format(
          end_time - start_time)


if __name__ == "__main__":
    main()
