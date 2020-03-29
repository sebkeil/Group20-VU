
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from scipy.stats import expon
from tpot import TPOTRegressor


# read in files
X_train = pd.read_csv('train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])

# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

y_train = y_train.values

# create the tpot configurations

tpot_config = {
    'sklearn.svm.SVR': {
        'kernel': ['rbf','poly'],
        'degree': [1,2,3,4,5,6,7,8,9,10],
        'gamma': [0.001,0.01,0.1,1,10],
        'C': [0.001,0.01,0.1,1]
    }}

tpot_optimizer = TPOTRegressor(generations=5, population_size=20, cv=5, verbosity=3,config_dict= 'TPOT light',n_jobs=1, max_time_mins=10, max_eval_time_mins=1)

# fit grid with data
tpot_optimizer.fit(X_train,y_train.ravel())

print(tpot_optimizer.pareto_front_fitted_pipelines_)
