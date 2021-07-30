#!/usr/bin/env python3

from collections import Counter
import urllib.request
from lxml import etree
import sklearn

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

alphabet="abcdefghijklmnopqrstuvwxyzäö-"
alphabet_set = set(alphabet)

# Returns a list of Finnish words
def load_finnish():
    finnish_url="https://www.cs.helsinki.fi/u/jttoivon/dap/data/kotus-sanalista_v1/kotus-sanalista_v1.xml"
    filename="data/kotus-sanalista_v1.xml"
    load_from_net=False
    if load_from_net:
        with urllib.request.urlopen(finnish_url) as data:
            lines=[]
            for line in data:
                lines.append(line.decode('utf-8'))
        doc="".join(lines)
    else:
        with open(filename, "rb") as data:
            doc=data.read()
    tree = etree.XML(doc)
    s_elements = tree.xpath('/kotus-sanalista/st/s')
    return list(map(lambda s: s.text, s_elements))

def load_english():
    with open("data/words", encoding="utf-8") as data:
        lines=map(lambda s: s.rstrip(), data.readlines())
    return list(lines)

def get_features(a):
    counts = np.zeros((a.shape[0], 29))

    for i, word in enumerate(a):
        c = Counter(word)
        for letter, count in c.items():
            letter_no = alphabet.index(letter)
            counts[i, letter_no] = count
    
    return counts

def contains_valid_chars(s):
    for letter in s:
        if letter not in alphabet:
            return False
    return True

def get_features_and_labels():
    finnish = load_finnish()
    english = load_english()

    lowercase_finnish = [word.lower() for word in finnish]
    filtered_finnish = list(filter(lambda word: contains_valid_chars(word), lowercase_finnish))

    no_nouns_english = list(filter(lambda word: word[0].islower(), english))
    lowercase_english = [word.lower() for word in no_nouns_english]
    filtered_english = list(filter(lambda word: contains_valid_chars(word), lowercase_english))

    feature_finnish = get_features(np.array(filtered_finnish))
    feature_english = get_features(np.array(filtered_english))

    target_finnish = np.zeros(len(filtered_finnish))
    target_english = np.ones(len(filtered_english))

    features = np.concatenate((feature_finnish, feature_english)) 
    target = np.concatenate((target_finnish, target_english)) 

    return features, target


def word_classification():
    features, target = get_features_and_labels()
    model = MultinomialNB()
    crossval = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(model, features, target, cv=crossval)


def main():
    print("Accuracy scores are:", word_classification())

if __name__ == "__main__":
    main()
