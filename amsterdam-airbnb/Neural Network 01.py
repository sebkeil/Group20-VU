from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_validate
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

# read in csv files
X_train = pd.read_csv('train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])

X_test = pd.read_csv('test.csv')
y_test = pd.read_csv('y_test.csv', names=['price'])

def build_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(16,)))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_squared_error'])
    return model

regressor = KerasRegressor(build_fn = build_model, batch_size = 10, epochs = 100)
error = cross_validate(estimator = regressor, X = X_train, y = y_train, scoring= ('neg_root_mean_squared_error', 'r2'), cv= 1, n_jobs = 1)

print(error)

# Fit the model
#history = model.fit(X_train,y_train, epochs=15, validation_split=0.3)

#plot metrics
#plt.plot(history.history['mean_squared_error'])
#plt.plot(history.history['mean_absolute_error'])
#plt.plot(history.history['mean_absolute_percentage_error'])
#plt.plot(history.history['cosine_proximity'])
#plt.show()




#score = model.evaluate(X_test, y_test, batch_size=44)

#rmse = round(sqrt(score[1]),2)
#print(rmse)