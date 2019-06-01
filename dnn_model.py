from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import keras


def create_model(input_dim, p=0.5, activation='sigmoid'):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim))
    model.add(Dropout(p))
    model.add(Dense(10))
    model.add(Dropout(p))
    model.add(Dense(1))
    model.add(Activation(activation))
    model.summary()
    return model
