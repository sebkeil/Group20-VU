from sklearn.ensemble import GradientBoostingRegressor
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
X_train = pd.read_csv('train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])
y_train = y_train.values

X_test = pd.read_csv('test.csv')
y_test = pd.read_csv('y_test.csv', names=['price'])

# drop features
X_train = X_train.drop(['bathrooms', 'bedrooms','guests_included','host_listings_count','instant_bookable_f','room_type_Private room'],axis=1)
X_test = X_test.drop(['bathrooms', 'bedrooms','guests_included','host_listings_count','instant_bookable_f','room_type_Private room'],axis=1)


# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a linear regression object: reg
gb = GradientBoostingRegressor(learning_rate= 0.1, n_estimators=200)

# fit  data
gb.fit(X_train,y_train.ravel())

# make predications
y_pred = gb.predict(X_test)

# compute metrics
regression_results(y_test,y_pred)
