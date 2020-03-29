from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

# read in csv files
X_train = pd.read_csv('train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])

X_test = pd.read_csv('test.csv')
y_test = pd.read_csv('y_test.csv', names=['price'])

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train,y_train, validation_split=0.3)