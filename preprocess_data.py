from collections import Counter
import numpy as np
import mlflow
import os
import sys

alphabet="abcdefghijklmnopqrstuvwxyzäö-"


def download_data(location, filename):
    with open(os.path.join(location, filename)) as data:
        return data.read()


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
    with mlflow.start_run():
        print('Uploading data from artifacts')

        data_location = sys.argv[2]
        finnish = download_data(data_location, 'finnish-list-raw/finnish_raw.txt')
        english = download_data(data_location, 'english-list-raw/english_raw.txt')

        print('Filtering & transforming data')
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

        np.savetxt('features.txt', features)
        np.savetxt('target.txt', target)

        mlflow.log_artifact('features.txt')
        mlflow.log_artifact('target.txt')


if __name__ == "__main__":
    get_features_and_labels()
