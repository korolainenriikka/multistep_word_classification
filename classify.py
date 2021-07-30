from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import model_selection


def word_classification():
    features, target = get_features_and_labels()
    model = MultinomialNB()
    crossval = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(model, features, target, cv=crossval)
