# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 01:02:38 2020

@author: Hamza-Acer
"""

import pandas as pd
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
# roc curve
from sklearn.metrics import roc_curve


# load the dataset
dataset = pd.read_csv('C:/Users/Hamza-Acer/Desktop/isdead.csv')
data_features = dataset.drop('is_dead', axis=1)
    
features = data_features.to_numpy()

target = dataset['is_dead']
target = target.to_numpy()

#we divide some data for later use
# caling works very good with neural network hence we try it...
from sklearn.preprocessing import MinMaxScaler
# load data
data = features
# create scaler
scaler = MinMaxScaler()
# fit scaler on data
scaler.fit(data)
# apply transform
normalized_features = scaler.transform(data)

X_train, X_test, Y_train, Y_test = train_test_split(normalized_features, target, test_size=0.1, random_state=42)

# define the keras model
model = Sequential()
model.add(Dense(32, input_dim=1, activation='relu'))
model.add(Dense(32, activation='relu'))
#model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam')

from matplotlib import pyplot
i = 0
while(i<24):
    # fit the keras model on the dataset
    history = model.fit(X_train[:,i], Y_train, validation_data=(X_test[:,i],Y_test) ,epochs=50, batch_size=100, verbose = 0)
    y_pred_keras = model.predict(X_test[:,0])
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_pred_keras)
    pyplot.plot(fpr_keras, tpr_keras, marker='.', label='ANN')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.title(dataset.columns.values[i])
    #this line saves the chart into a file...
    #pyplot.savefig('C:/Users/Hamza-Acer/Desktop/ROC/'+dataset.columns.values[i]+'.png')
    pyplot.show()
    i+=1