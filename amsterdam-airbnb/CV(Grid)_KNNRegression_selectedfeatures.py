from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# read in files
X_train = pd.read_csv('train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])

# drop features
X_train = X_train.drop(['bathrooms', 'bedrooms','guests_included','host_listings_count','instant_bookable_f','room_type_Private room'],axis=1)

# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Create a linear regression object: reg
knn = KNeighborsRegressor()

# instantiate the search range
k_range = range(1,31)

# create the parameter grid
param_grid = dict(n_neighbors=k_range)
print(param_grid)

# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=5, scoring=('r2', 'neg_root_mean_squared_error'), refit=False)

# fit grid with data
grid.fit(X_train,y_train)

# view the complete results
df = pd.DataFrame(grid.cv_results_)[['params', 'mean_test_r2', 'mean_test_neg_root_mean_squared_error']]


df.to_csv('knn_regression_selectedfeatures.csv')




grid_mean_scores = -1 * grid.cv_results_['mean_test_neg_root_mean_squared_error']

#plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('RMSE')
plt.show()
