from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

import mlflow

# TODO: add some kind of hyperparam for this to demonstrate hyperparam passing in a pipeline
# smthn like how big part of the data should be used

def word_classification(features, target):
    print('Starting model run')
    model = MultinomialNB()
    crossval = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

    accuracy_scores = cross_val_score(model, features, target, cv=crossval)
    mlflow.log_metric('accuracies', accuracy_scores)
    mlflow.sklearn.load_model(model)


if __name__ == "__main__":
    word_classification()
