from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras import regularizers
import keras


def create_model(input_dim, p=0.5, activation='sigmoid'):
    model = Sequential()
    model.add(Dense(100, kernel_regularizer=regularizers.l2(0.01), input_dim=input_dim))
    model.add(Dropout(p))
    model.add(Dense(100, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(p))
    model.add(Dense(1))
    model.add(Activation(activation))
    model.summary()
    return model
