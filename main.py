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

X, Y, x_test, y_test = dataset.dataset()

Y = to_categorical(Y, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

mbs = 16
epochs = 100
avg_auc = 0
n = 4
optimizer = Adadelta()
kf = KFold(n_splits=n)
for train_index, valid_index in kf.split(X):
    x_train = X[train_index]
    y_train = Y[train_index]

    x_valid = X[valid_index]
    y_valid = Y[valid_index]

    model = create_model(input_dim=x_train[0].shape[0])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    history = model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              epochs=epochs,
              verbose=1,
              batch_size=mbs)
    predictions = model.predict_proba(x_valid)
    #predictions = predictions[:, 1]
    auc = roc_auc_score(y_valid, predictions)
    print("ROC AUC:", auc / n)
    avg_auc += auc

print("Average AUC ROC:", avg_auc)
print("testing data")
print(model.evaluate(x_test, y_test))
score = model.predict_proba(x_test)
print(roc_auc_score(y_test, score))

#utility.plot_loss(history.history, 'Loss per Epoch', '.', 1)
