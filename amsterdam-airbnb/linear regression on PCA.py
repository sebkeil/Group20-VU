import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from math import sqrt

def regression_results(y_test, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_test, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred)
    mse=metrics.mean_squared_error(y_test, y_pred)
    #mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_test, y_pred)
    r2=metrics.r2_score(y_test, y_pred)

    print('explained_variance: ', round(explained_variance,2))
   # print('mean_squared_log_error: ', round(mean_squared_log_error,2))
    print('r2: ', round(r2,2))
    print('mean_absolute_error: ', round(mean_absolute_error,2))
    print('MSE: ', round(mse,2))
    print('RMSE: ', round(sqrt(mse),2))
    print('median_absolute_error:', round(median_absolute_error,2))

# read in csv files
X_train = pd.read_csv('pca_df_X_train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])

X_test = pd.read_csv('pca_df_X_test.csv')
y_test = pd.read_csv('y_test.csv', names=['price'])

# drop features
X_train = X_train.drop(['Unnamed: 0','PC7'], axis=1)
X_test = X_test.drop(['Unnamed: 0','PC7'], axis=1)

print(X_train.head())

# initialize and fit regressor
lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train, y_train)

# make predications
y_pred = lin_reg.predict(X_test)

# compute metrics
regression_results(y_test,y_pred)
print(lin_reg.coef_)
print(dict(zip(X_train.columns, abs(lin_reg.coef_[0]).round(2))))