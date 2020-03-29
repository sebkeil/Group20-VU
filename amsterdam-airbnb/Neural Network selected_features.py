from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import sqrt


# read in csv files
X_train = pd.read_csv('train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])

X_test = pd.read_csv('test.csv')
y_test = pd.read_csv('y_test.csv', names=['price'])


# drop features

X_train = X_train.drop(['bathrooms', 'bedrooms','guests_included','host_listings_count','instant_bookable_f','room_type_Private room'],axis=1)
X_test = X_test.drop(['bathrooms', 'bedrooms','guests_included','host_listings_count','instant_bookable_f','room_type_Private room'],axis=1)

# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# build model

model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(10,)))
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

scores = model.evaluate(x=X_test, y=y_test)
rmse = round(sqrt(scores[1]),2)
print(rmse)

