# Jeff Lund
# Runner for deep neural network on WNV dataset
# python3 dnn.py

import os
import sys
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np

from dnn_model import create_model
import dataset
import utility


X, Y, x_test, y_test = dataset.dataset(pca=False, impute=True)

metrics = []

sgd = SGD(lr=0.01, momentum=0.90)
adam = Adam()
ada = Adadelta()
optimizers = [('Adadelta', ada), ('Adam', adam), ('SGD', sgd)]

for name, opt in optimizers:
    print("Using optimizer:", name)
    mbs = 32
    avg_auc = 0
    n_folds = 5
    models = []
    histories = []
    aucs = []
    kf = KFold(n_splits=n_folds, shuffle=True)
    for i, (train_index, valid_index) in enumerate(kf.split(X)):
        start = time()
        print("Fold", i)
        model = create_model(input_dim=X[0].shape[0])
        model.compile(optimizer=opt, loss='binary_crossentropy')
        X, Y = dataset.shuffle(X, Y)
        x_train = X[train_index]
        y_train = Y[train_index]
        x_valid = X[valid_index]
        y_valid = Y[valid_index]
        history = model.fit(x_train, y_train,
                  validation_data=(x_valid, y_valid),
                  epochs=200,
                  verbose=0,
                  batch_size=mbs,
                  callbacks=[EarlyStopping(patience=15, restore_best_weights=True)])
        predictions = model.predict_proba(x_valid)
        auc = roc_auc_score(y_valid, predictions)
        print("ROC AUC:", auc)
        avg_auc += auc
        models.append(model)
        histories.append(history)
        aucs.append(auc)
        end = time()
        print("Total time", end - start)
    index = np.argmax(aucs) # find best model from k folds
    model = models[index]
    print("Average AUC ROC:", avg_auc / n_folds)
    print("testing data")
    x_prob = model.predict_proba(x_train)
    y_prob = model.predict_proba(x_test)
    r_test = roc_auc_score(y_test, y_prob)
    print("Test AUC ROC:", r_test)
    utility.roc(y_train, x_prob, y_test, y_prob, name)
    utility.plot_loss(histories, '{} Model Loss per Epoch'.format(name))
    metrics.append((name, r_test))

print(metrics)
