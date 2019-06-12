import dataset
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_selection import SelectPercentile, SelectFromModel, SelectFpr, SelectFdr


class LDA_QDA:
    def __init__(self):
        self.x_train, self.y_train, self.x_test, self.y_test = dataset.dataset2()

    def run_model(self, model, name, x_train=None, x_test=None):
        """
        Fit an LDA or QDA model, calculate confidence scores and the ROC-AUC score for the training data and test data,
        and plot the ROC curves for the training data and test data.

        :param model: LDA or QDA model
        :param name: experiment name (for roc-auc plot filename)
        :param x_train: optional parameter to use if experiment uses a reduced dataset
        :param x_test: optional parameter to use if experiment uses a reduced dataset
        :return:
        """
        if x_train is None:
            x_train = self.x_train
        if x_test is None:
            x_test = self.x_test

        model.fit(x_train, self.y_train)

        train_scores = model.decision_function(x_train)
        train_roc_auc = roc_auc_score(self.y_train, train_scores)
        print("ROC-AUC score on training data: %s" % train_roc_auc)

        test_scores = model.decision_function(x_test)
        test_roc_auc = roc_auc_score(self.y_test, test_scores)
        print("ROC-AUC score on test data: %s\n" % test_roc_auc)

        self.plot_roc_curve(train_scores, test_scores, name)

    def select_percentile(self):
        models = {"lda": LDA(), "qda": QDA()}
        for name in models:
            print(name)
            percentiles = [50, 60, 70]
            for p in percentiles:
                print("percentile selection (%s):" % p)
                models[name].fit(self.x_train, self.y_train)

                m = SelectPercentile(percentile=p)
                x_train = m.fit_transform(self.x_train, self.y_train)
                x_test = m.transform(self.x_test)

                self.run_model(models[name], "%s_percentile_selection_%s" % (name, p), x_train, x_test)

    def fdr_feature_selection(self):
        alphas = [.1, .05, .01, .001, .0001]
        for a in alphas:
            model = LDA()
            print("LDA with fpr feature selection (%s):" % a)
            model.fit(self.x_train, self.y_train)

            m = SelectFdr(alpha=a)
            x_train = m.fit_transform(self.x_train, self.y_train)
            x_test = m.transform(self.x_test)

            self.run_model(model, "fpr_feature_selection_%s" % a, x_train, x_test)

    def fpr_feature_selection(self):
        alphas = [.1, .05, .01, .001, .0001]
        for a in alphas:
            model = LDA()
            print("LDA with fpr feature selection (%s):" % a)
            model.fit(self.x_train, self.y_train)

            m = SelectFpr(alpha=a)
            x_train = m.fit_transform(self.x_train, self.y_train)
            x_test = m.transform(self.x_test)

            self.run_model(model, "fpr_feature_selection_%s" % a, x_train, x_test)

    def select_from_model(self):
        thresholds = [.1, .2, .3, .4, .5]
        for t in thresholds:
            print("LDA with feature selection (%s):" % t)
            model = LDA()
            model.fit(self.x_train, self.y_train)

            m = SelectFromModel(model, threshold=t, prefit=True)
            x_train = m.transform(self.x_train)
            x_test = m.transform(self.x_test)
            self.run_model(model, "lda_feature_selection_%s" % t, x_train, x_test)

    def lda(self, solver="svd", prior=None):
        """
        Create an LDA model based on the specified parameters.

        :param solver: "svd", "lsqr", or "eigen" - solving method used by the model
        :param prior: optional list of prior values for the two classes to use in the model (ex: [.9, .1])
        :return:
        """
        print("LDA:")
        model = LDA(solver=solver, priors=prior)
        self.run_model(model, "lda_%s_%s" % (solver, prior))

    def qda(self, prior=None):
        """
        Create a QDA model.

        :param prior: optional list of prior values for the two classes to use in the model (ex: [.9, .1])
        :return:
        """
        print("QDA:")
        model = QDA(priors=prior)
        self.run_model(model, "qda_%s" % prior)

    def run_experiments(self):
        """
        Run default LDA and QDA models on the data, and run experiments with different parameter values.

        :return:
        """
        # Default
        self.lda()
        self.qda()

        # Experiment 1: different solver methods with the LDA
        solver_options = ["lsqr", "svd", "eigen"]
        for s in solver_options:
            print("Solver: %s" % s)
            self.lda(solver=s)

        # Experiment 2: different prior values with the LDA and QDA
        prior_options = [None, [.9, .1], [.95, .05], [.99, .01]]
        for p in prior_options:
            print("Prior: %s" % p)
            self.lda(prior=p)
            self.qda(prior=p)

        # Experiment 3: feature selection
        self.select_from_model()
        self.fdr_feature_selection()
        self.fpr_feature_selection()
        self.select_percentile()

    def plot_roc_curve(self, train_scores, test_scores, filename):
        """
        Creates the ROC curve on the results of the training data and test data.
        (Modified the example from https://scikit-learn.org/0.15/auto_examples/plot_roc.html)

        :param train_scores: result of decision_function for the model with training data
        :param test_scores: result of decision_function for the model with test data
        :param filename: filename for plot
        :return:
        """
        # Calculate false positive rates and true positive rates for ROC curves
        train_fpr, train_tpr, _ = roc_curve(self.y_train, train_scores)
        test_fpr, test_tpr, _ = roc_curve(self.y_test, test_scores)

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
        plt.close()


if __name__ == "__main__":
    LDA_QDA().run_experiments()
