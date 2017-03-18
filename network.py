import util
import math
import random
import numpy as np
import matplotlib as plot
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense,Dropout,GRU,Reshape

file_name = 'dataset.csv'
leng = 12

def main():

    #Building network
    print("Building net..",end="")
    net = Sequential()
    net.add(Dense(leng,input_dim=leng,activation='linear'))
    net.add(Reshape((1,leng)))
    net.add(GRU(30,activation='relu',return_sequences=True))
    net.add(Dropout(0.3))
    net.add(GRU(49,activation='sigmoid',return_sequences=False))
    net.add(Dense(1,activation='sigmoid'))
    net.compile(optimizer=SGD(lr=.01,momentum=.9,nesterov=True),loss='mse')
    print("done!")

    #Data
    print("Loading data...",end="")
    f            = open(file_name,'r')
    data,labels  = util.loadData(f)
    data         = util.reduceMatRows(data)
    labels,m1,m2 = util.reduceVector(labels,getVal=True)
    print("{} chunks loaded!\nTraining...".format(len(labels)),end="")

    #Training dnn
    net.fit(data,labels,nb_epoch=250)

    print("trained!")

    for i in range(len(data)): #Train all over the dataset to give weights to the RNN unit
        x         = np.array(data[i]).reshape(1,12)       #Reshape input data to forward it inside the dnn
        predicted = util.augmentValue(net.predict(x)[0],m1,m2) #Denormalize data 
        real      = util.augmentValue(labels[i],m1,m2)    #Denormalize data
        lhood     = abs( math.sqrt((real-predicted)**2))  #Calculate distance between predicted and real
        if i > (len(data)-20):print("Real:{} Predicted:{}  likehood({})".format(real,predicted,  ))

if __name__ == '__main__':
    main()