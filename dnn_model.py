from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
from keras import regularizers
import keras

def create_model(input_dim, p=0.5, activation='softmax'):

    model = Sequential()

    model.add(Dense(32, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(p))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(p))

    model.add(Dense(2))
    model.add(Activation(activation))
    #model.summary()

    return model
