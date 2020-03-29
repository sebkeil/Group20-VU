from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
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
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_validate(reg, X_train, y_train, cv=5, scoring=('r2', 'neg_root_mean_squared_error'))

# Print the 5-fold cross-validation scores

#print(cv_scores)

print("Average 5-Fold CV Score (R2): {}".format(round(np.mean(cv_scores['test_r2']),4)))
print("Average 5-Fold CV Score (RMSE): {}".format(round(np.mean(cv_scores['test_neg_root_mean_squared_error']),2)))

