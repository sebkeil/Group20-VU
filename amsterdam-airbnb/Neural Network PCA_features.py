from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_validate
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

# read in files
X_train = pd.read_csv('pca_df_X_train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])


X_test = pd.read_csv('pca_df_X_test.csv')
y_test = pd.read_csv('y_test.csv', names=['price'])


# drop features
X_train = X_train.drop(['Unnamed: 0','PC7'], axis=1)
X_test = X_test.drop(['Unnamed: 0','PC7'], axis=1)

# build model

model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(6,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_squared_error'])


# Fit the model
history = model.fit(X_train,y_train, epochs=20, validation_split=0.3)

'''
#plot metrics
plt.plot(history.history['mean_squared_error'])
#plt.plot(history.history['mean_absolute_error'])
#plt.plot(history.history['mean_absolute_percentage_error'])
#plt.plot(history.history['cosine_proximity'])
plt.show()
'''



score = model.evaluate(X_test, y_test, batch_size=44)

rmse = round(sqrt(score[1]),2)
print(rmse)