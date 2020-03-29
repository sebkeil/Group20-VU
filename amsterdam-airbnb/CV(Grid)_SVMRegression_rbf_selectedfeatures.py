
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# read in files
X_train = pd.read_csv('train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])
y_train = y_train.values


# drop features

X_train = X_train.drop(['bathrooms', 'bedrooms','guests_included','host_listings_count','instant_bookable_f','room_type_Private room'],axis=1)


# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Create a linear regression object: reg
svr = SVR(C=10)

gamma_range = [0.02, 0.03,0.04,0.05,0.06]

# create the parameter grid
param_grid = dict(gamma=gamma_range)

# instantiate the grid

grid = GridSearchCV(svr, param_grid, cv=5, scoring=('r2', 'neg_root_mean_squared_error'), refit=False)

# fit grid with data
grid.fit(X_train,y_train.ravel())

# view the complete results
df = pd.DataFrame(grid.cv_results_)[['params', 'mean_test_r2','mean_test_neg_root_mean_squared_error']]


df.to_csv('svr_regression_rbf_selectedfeatures.csv')


grid_mean_scores = -1 * grid.cv_results_['mean_test_neg_root_mean_squared_error']

#plot the results
plt.plot(gamma_range, grid_mean_scores)
plt.xlabel('Value of gamma')
plt.ylabel('RMSE')
plt.show()