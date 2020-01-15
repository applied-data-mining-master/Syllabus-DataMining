from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def own_cross_val(classifier, X, y):
    skfolds = StratifiedKFold(n_splits=3, random_state=42)

    result = []
    for train_index, test_index in skfolds.split(X, y):
        clone_clf = clone(classifier)
        X_train_folds = X[train_index]
        y_train_folds = (y[train_index])
        X_test_fold = X[test_index]
        y_test_fold = (y[test_index])

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        result.append(n_correct / len(y_pred))
    return result
