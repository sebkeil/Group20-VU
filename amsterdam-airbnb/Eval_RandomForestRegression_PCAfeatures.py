from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

    print('r2: ', round(r2,4))
    print('RMSE: ', round(sqrt(mse),2))


# read in files
X_train = pd.read_csv('pca_df_X_train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])
y_train = y_train.values

X_test = pd.read_csv('pca_df_X_test.csv')
y_test = pd.read_csv('y_test.csv', names=['price'])

# drop features
X_train = X_train.drop(['Unnamed: 0','PC7'], axis=1)
X_test= X_test.drop(['Unnamed: 0','PC7'], axis=1)

# Create a linear regression object: reg
rf = RandomForestRegressor(n_estimators=200)

# fit  data
rf.fit(X_train,y_train.ravel())

# make predications
y_pred = rf.predict(X_test)

# compute metrics
regression_results(y_test,y_pred)
