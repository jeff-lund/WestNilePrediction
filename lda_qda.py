import dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import roc_auc_score


def run_model(model):
    model.fit(x_train, y_train)

    train_scores = model.score(x_train, y_train)
    print("Mean accuracy on training data: %s" % train_scores)

    train_decisions = model.decision_function(x_train)
    roc_auc = roc_auc_score(y_train, train_decisions)
    print("ROC-AUC score on training data: %s" % roc_auc)

    test_scores = model.score(x_test, y_test)
    print("Mean accuracy on test data: %s" % test_scores)

    test_decisions = model.decision_function(x_test)
    roc_auc = roc_auc_score(y_test, test_decisions)
    print("ROC-AUC score on test data: %s" % roc_auc)


def lda():
    print("LDA:")
    model = LDA()
    run_model(model)


def qda():
    print("\nQDA:")
    model = QDA()
    run_model(model)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = dataset.dataset2()
    lda()
    qda()
