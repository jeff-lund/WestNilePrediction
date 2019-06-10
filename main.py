import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import KFold
import sklearn as sk
import numpy as np
from dnn_model import create_model
import dataset
import utility

X, Y, x_test, y_test = dataset.dataset(pca=True)

Y = to_categorical(Y, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
#optimizers = [Adadelta, Adam, SGD, Adagrad, RMSprop]
#adam = Adam(lr=0.001)
#sgd = SGD(lr=0.001, momentum=0.9)
#ada = Adagrad(lr=0.001)
layers = [32]
ls = 32
lrs = [0.0001, 0.001]
#, ('sgd', sgd), ('adagrad', ada)]
metric = []

epochs = [100]
adam = Adam(lr=0.00001)
sgd = SGD(lr=0.01, momentum=0.90)
optimizers = [('sgd', sgd)]
for name, opt in optimizers:
    for e in epochs:
        print("Using optimizer:", name)
        print("Epochs", e)
        mbs = 16
        avg_auc = 0
        n = 5
        #model = create_model(input_dim=X[0].shape[0], layer_size=ls)
        #model.compile(optimizer=opt, loss='categorical_crossentropy')
        models = []
        aucs = []
        kf = KFold(n_splits=n, shuffle=True)
        for i, (train_index, valid_index) in enumerate(kf.split(X)):
            print("Fold", i)
            X, Y = dataset.shuffle(X, Y)
            x_train = X[train_index]
            y_train = Y[train_index]
            model = create_model(input_dim=X[0].shape[0], layer_size=ls)
            model.compile(optimizer=opt, loss='categorical_crossentropy')
            x_valid = X[valid_index]
            y_valid = Y[valid_index]
            history = model.fit(x_train, y_train,
                      validation_data=(x_valid, y_valid),
                      epochs=e,
                      verbose=0,
                      batch_size=mbs)
            predictions = model.predict_proba(x_valid)
            auc = roc_auc_score(y_valid, predictions)
            print("ROC AUC:", auc)
            avg_auc += auc
            models.append(model)
            aucs.append(auc)
        index = np.argmax(aucs)
        model2 = models[index]
        print("Average AUC ROC:", avg_auc / n)
        print("testing data")
        print(model.evaluate(x_test, y_test))
        score = model.predict_proba(x_test)
        r_test = roc_auc_score(y_test, score)
        print(r_test)

        print(model2.evaluate(x_test, y_test))
        score = model2.predict_proba(x_test)
        r_test = roc_auc_score(y_test, score)
        print(r_test)

        metric.append((name, e, r_test))
print(metric)
#utility.plot_loss(history.history, 'Loss per Epoch', '.', 1)
