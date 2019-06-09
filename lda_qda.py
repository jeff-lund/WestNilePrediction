import dataset
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import roc_auc_score, roc_curve


def run_model(model, name):
    """
    Fit an LDA or QDA model, calculate confidence scores and the ROC-AUC score for the training data and test data,
    and plot the ROC curves for the training data and test data.

    :param model: LDA or QDA model
    :param name: experiment name (for roc-auc plot filename)
    :return:
    """
    model.fit(x_train, y_train)

    train_scores = model.decision_function(x_train)
    train_roc_auc = roc_auc_score(y_train, train_scores)
    print("ROC-AUC score on training data: %s" % train_roc_auc)

    test_scores = model.decision_function(x_test)
    test_roc_auc = roc_auc_score(y_test, test_scores)
    print("ROC-AUC score on test data: %s" % test_roc_auc)

    plot_roc_curve(train_scores, test_scores, name)


def lda(solver="svd", prior=None):
    """
    Create an LDA model based on the specified parameters.

    :param solver: "svd", "lsqr", or "eigen" - solving method used by the model
    :param prior: optional list of prior values for the two classes to use in the model (ex: [.9, .1])
    :return:
    """
    print("LDA:")
    model = LDA(solver=solver, priors=prior)
    run_model(model, "lda_%s_%s" % (solver, prior))


def qda(prior=None):
    """
    Create a QDA model.

    :param prior: optional list of prior values for the two classes to use in the model (ex: [.9, .1])
    :return:
    """
    print("\nQDA:")
    model = QDA(priors=prior)
    run_model(model, "qda_%s" % prior)


def run_experiments():
    """
    Run default LDA and QDA models on the data, and run experiments with different parameter values.

    :return:
    """
    # Default
    lda()
    qda()

    # Experiment 1: different solver methods with the LDA
    solver_options = ["lsqr", "svd", "eigen"]
    for s in solver_options:
        print("\nSolver: %s" % s)
        lda(solver=s)

    # Experiment 2: different prior values with the LDA and QDA
    prior_options = [None, [.9, .1], [.99, .01]]
    for p in prior_options:
        print("\nPrior: %s" % p)
        lda(prior=p)
        qda(prior=p)


def plot_roc_curve(train_scores, test_scores, filename):
    """
    Creates the ROC curve on the results of the training data and test data.
    (Modified the example from https://scikit-learn.org/0.15/auto_examples/plot_roc.html)

    :param train_scores: result of decision_function for the model with training data
    :param test_scores: result of decision_function for the model with test data
    :param filename: filename for plot
    :return:
    """
    # Calculate false positive rates and true positive rates for ROC curves
    train_fpr, train_tpr, _ = roc_curve(y_train, train_scores)
    test_fpr, test_tpr, _ = roc_curve(y_test, test_scores)

    # Plot the ROC curves
    plt.figure()
    plt.plot(train_fpr, train_tpr, label='Training', color='teal')
    plt.plot(test_fpr, test_tpr, label='Test', color='tab:pink')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('%s.png' % filename)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = dataset.dataset2()
    run_experiments()
