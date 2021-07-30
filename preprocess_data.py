from collections import Counter
import numpy as np
import mlflow

alphabet="abcdefghijklmnopqrstuvwxyzäö-"

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


def get_features_and_labels(finnish, english):
    with mlflow.start_run():
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

        mlflow.log_artifacts(features)
        mlflow.log_artifacts(target)


if __name__ == "__main__":
    get_features_and_labels()
