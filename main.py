import keras
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
from sklearn.metrics import roc_auc_score
from dnn_model import create_model
import dataset

x_train, y_train, x_test, y_test = dataset.dataset()
model = create_model(x_train[0].shape[0], p=0.6)
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['mse', 'accuracy'])
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=500,
          batch_size=128)

print("training data")
print(model.evaluate(x_train, y_train))
score = model.predict(x_train)
print(score[100:])
print(roc_auc_score(y_train, score))

print()
print("*" * 40)
print()

print("testing data")
print(model.evaluate(x_test, y_test))
score = model.predict(x_test)
print(score[100:])
print(roc_auc_score(y_test, score))
