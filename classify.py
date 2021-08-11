from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

import numpy as np
import mlflow
import sys
import os


def word_classification():
    print('Loading preprocessed data from artifacts')

    data_location = sys.argv[2]
    features = np.genfromtxt(os.path.join(data_location, 'features.txt'))
    target = np.genfromtxt(os.path.join(data_location, 'target.txt'))

    print('Starting model run')
    model = MultinomialNB()
    crossval = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

    accuracy_scores = cross_val_score(model, features, target, cv=crossval)
    for i, accuracy in enumerate(accuracy_scores):
        mlflow.log_metric('accuracy' + str(i), accuracy)
    mlflow.sklearn.log_model(model, 'model')


if __name__ == "__main__":
    word_classification()
