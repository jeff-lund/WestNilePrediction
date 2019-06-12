# Jeff Lund
# graphing functions for use with DNN model

import matplotlib.pyplot as plt
import sklearn
import os

def plot_loss(history, title):
    '''
    creates a plot of loss vs epochs
    history is the history field from the history object
    returned from fitting a model
    '''
    for i, h in enumerate(history):
        x_axis = [x for x in range(len(h.history['loss']))]
        plt.plot(x_axis, h.history['loss'], label="Training {}".format(i+1))
        plt.plot(x_axis, h.history['val_loss'], label="Validation {}".format(i+1))
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    sv_title = title.replace(' ', '_')
    sv_title = sv_title
    plt.savefig(sv_title + '_loss.png', format='png', bbox_inches='tight')
    plt.close()

def roc(y_train, x_prob, y_test, y_prob, title):
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_train, x_prob, pos_label=1)
    plt.plot(fpr, tpr, label='Training')
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_prob, pos_label=1)
    plt.plot(fpr, tpr, label='Validation')
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive rate")
    plt.title("{} ROC Curve".format(title))
    plt.legend()
    plt.savefig('{}_roc.png'.format(title), format='png')
    plt.close()
