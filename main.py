import keras
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
from sklearn.metrics import roc_auc_score
from dnn_model import create_model
import dataset

x_train, y_train, x_test = dataset.dataset()
model = create_model(x_train[0].shape[0], p=0.6)
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['mse', 'accuracy'])
model.fit(x_train, y_train, epochs=6000, batch_size=5000)
print(model.evaluate(x_train, y_train))
score = model.predict_on_batch(x_train[:32])
print(score)
