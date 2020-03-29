
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# read in files
X_train = pd.read_csv('train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])

# drop features

X_train = X_train.drop(['bathrooms', 'bedrooms','guests_included','host_listings_count','instant_bookable_f','room_type_Private room'],axis=1)

# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Create a linear regression object: reg
ridge = Ridge()

# instantiate the search range
alpha_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,100]

# create the parameter grid
param_grid = dict(alpha=alpha_range)
print(param_grid)

# instantiate the grid

grid = GridSearchCV(ridge, param_grid, cv=5, scoring=('r2', 'neg_root_mean_squared_error'), refit=False)

# fit grid with data
grid.fit(X_train,y_train)

# view the complete results
df = pd.DataFrame(grid.cv_results_)[['params', 'mean_test_r2','mean_test_neg_root_mean_squared_error']]
print(df)

#df.to_csv('knn_regression_fullfeatures.csv')


grid_mean_scores = -1 * grid.cv_results_['mean_test_neg_root_mean_squared_error']

#plot the results
plt.plot(alpha_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('RMSE')
plt.show()
