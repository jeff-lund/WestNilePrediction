# Jeff Lund
# Keras based DNN model for WNV Classification
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
from keras.regularizers import l2, l1_l2
import keras

def create_model(input_dim, p=0.5, layer_size=32, activation='sigmoid'):

    model = Sequential()

    model.add(Dense(layer_size, kernel_regularizer=l2(0.001), input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(p))

    model.add(Dense(layer_size, kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(p))

    #model.add(Dense(layer_size, kernel_regularizer=l2(0.001)))
    #model.add(Activation('relu'))
    #model.add(Dropout(p))

    model.add(Dense(1))
    model.add(Activation(activation))

    #model.summary()

    return model
