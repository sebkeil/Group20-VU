import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
from math import sqrt


X_train = pd.read_csv('train.csv', skiprows=1)
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('test.csv', skiprows=1)
y_test = pd.read_csv('y_test.csv')

# some exploratory data analyses

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#print(X_train.info())
#print(X_train.describe())
#print(X_train.head())

# fit Regressor to training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(y_pred)

# compute metrics
mse = mean_squared_error(y_test, y_pred)
mean_error = sqrt(mse)
print(mse)

print("The average prediction error is %.2f " %mean_error)


def regression_results(y_test, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_test, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred)
    mse=metrics.mean_squared_error(y_test, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_test, y_pred)
    r2=metrics.r2_score(y_test, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(sqrt(mse),4))

regression_results(y_test, y_pred)


