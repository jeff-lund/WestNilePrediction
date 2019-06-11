"""
West Nile Prediction using a Random Forest Classifier
__author__ : Colleen Rooney
"""

import random
import numpy as np
from sklearn import ensemble, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import dataset

def plot_roc(clf, file_name='temp.png'):
    predictions = clf.predict_proba(X)[:,1]
    tr_fpr, tr_tpr, thresholds = metrics.roc_curve(Y, predictions, pos_label=1)
    predictions = clf.predict_proba(x_test)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
    
    auc = metrics.auc(fpr, tpr)
    print(auc)

    plt.plot(fpr, tpr, color='magenta', label='Test')
    plt.plot(tr_fpr, tr_tpr, color='teal', label='Training')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Random Forest ROC Curve')
    plt.legend()
    plt.savefig(file_name)
    plt.show()


def experiment(test_list, classifiers, xlabel='', plot_title='',
        file_name='temp.png', plot=True):
    test_auc = []
    train_auc = []
    best = 1
    for clf in classifiers:
        clf.fit(X, Y)

        predictions = clf.predict_proba(X)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(Y, predictions,
                pos_label=1)
        tr_auc = metrics.auc(fpr, tpr)
        print(tr_auc)
        print(clf.n_estimators)
        train_auc.append(tr_auc)

        predictions = clf.predict_proba(x_test)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions,
                pos_label=1)
        auc = metrics.auc(fpr, tpr)
        test_auc.append(auc)
        print(auc)
        diff = tr_auc - auc
        print(diff)
        if diff < best and auc > 0.6:
            best_classifier = clf
        print("-------------")

    if plot:
        plt.plot(test_list, test_auc, 'y', label='Test AUC')
        plt.plot(test_list, train_auc, 'c', label='Train AUC')
        plt.ylabel('AUC')
        plt.xlabel(xlabel)
        plt.title(plot_title)
        plt.legend()
        plt.savefig(file_name)
        plt.show()
    return best_classifier


#-----------------------------------------------------------------------------
# Which tests to run
#-----------------------------------------------------------------------------
HYPER_PARAMS = False
ROC = True
GRID_SEARCH = False

#-----------------------------------------------------------------------------
# Load data
#-----------------------------------------------------------------------------
X, Y, x_test, y_test = dataset.dataset()
features = X.shape[1]


#-----------------------------------------------------------------------------
# Graph effect of individual hyperparameters
#-----------------------------------------------------------------------------
if HYPER_PARAMS:
    test_max_depth = list(range(1, 50, 2))
    test_estimators = list(range(1, 500, 10)) + list(range(500, 5000, 200))
    test_max_features = list(range(1, features, 2))
    test_min_samples_split = np.linspace(0.01, 1, 20)
    test_min_samples_leaf = np.linspace(0.01, 0.5, 20)
    clfs = []
    for n in test_max_depth:
        clfs.append(ensemble.RandomForestClassifier(n_jobs=4,  max_depth=n))
    experiment(test_max_depth, clfs, "Max Depth", "Effect of Max Depth on AUC",
            "max_depth_auc.png")

    clfs = []
    for n in test_max_features:
        clfs.append(ensemble.RandomForestClassifier(n_jobs=4,  max_features=n))
    experiment(test_max_features, clfs, "Max Features",
            "Effect of Max Features on AUC", "max_features_auc.png")

    clfs = []
    for n in test_min_samples_split:
        clfs.append(ensemble.RandomForestClassifier(n_jobs=4,
            min_samples_split=n))
    experiment(test_min_samples_split, clfs, "Percent Min Samples Split",
            "Effect of Min Samples Split on AUC", "min_samples_split_auc.png")

    clfs = []
    for n in test_min_samples_leaf:
        clfs.append(ensemble.RandomForestClassifier(n_jobs=4,
            min_samples_leaf=n))
    experiment(test_min_samples_leaf, clfs, "Percent Min Samples Leaf",
            "Effect of Min Samples Leaf on AUC", "min_samples_leaf_auc.png")

    clfs = []
    for n in test_estimators:
        clfs.append(ensemble.RandomForestClassifier(n_jobs=4,  n_estimators=n))
    experiment(test_estimators, clfs, "Number of Estimators",
            "Effect of Number of Estimators on AUC", "n_estimators_auc.png")

#-----------------------------------------------------------------------------
# Plot ROC with "out of the box" classifier
#-----------------------------------------------------------------------------
if ROC:
    clf = ensemble.RandomForestClassifier(n_jobs=4)
    clf.fit(X, Y)
    plot_roc(clf, 'naive_RF_ROC.png')


#-----------------------------------------------------------------------------
# Use GridSearchCV to select hyperparameters and plot ROC
#-----------------------------------------------------------------------------
if GRID_SEARCH:
    params = {
            'max_depth': [1, 2, 3, 4, 5],
            'min_samples_leaf': [0.2, 0.23, 0.26, 0.29],
            'min_samples_split': [0.4, 0.5, 0.6]
    }

    rf = ensemble.RandomForestClassifier()
    clf = GridSearchCV(estimator=rf, param_grid=params, scoring='roc_auc',
            cv=3, n_jobs=4)
    clf.fit(X, Y)
    print(clf.best_params_)
    print(clf.best_score_)

    best_model = clf.best_estimator_
    plot_roc(best_model, 'GS_RF_ROC.png')

