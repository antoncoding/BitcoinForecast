from keras.layers import Dense, Dropout, GRU, Reshape, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

def get_model(opt, input_shape):
    print("Building net..")
    model = Sequential()
    model.add(GRU(60, recurrent_dropout = 0.4, activation='relu',return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(GRU(100, recurrent_dropout = 0.4, activation='relu',return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(100, recurrent_dropout = 0.4, activation='relu',return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(3,activation='sigmoid'))
    model.add(Dropout(0.4))
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

    return model

