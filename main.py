import keras
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
from sklearn.metrics import roc_auc_score
from dnn_model import create_model
import dataset

x_train, y_train, x_test = dataset.dataset()
model = create_model(x_train[0].shape[0], p=0.6)
model.compile(optimizer=Adam(lr=0.0001),
              loss='mse',
              metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size=4)
print(model.evaluate(x_train, y_train))
score = model.predict_on_batch(x_train[:32])
print(score)
