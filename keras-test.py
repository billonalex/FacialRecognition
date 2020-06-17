# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:55:31 2020

@author: Alexandre
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
model = Sequential()

model.add(Conv2D(32, (3, 3)))
#model.add(Dense(units=16, activation='relu'))
#model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('OK')

#model.fit(X, Y, nb_epoch=200, validation_split=.20)