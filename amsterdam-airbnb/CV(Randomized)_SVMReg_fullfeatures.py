from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
from scipy.stats import expon
import numpy as np
import matplotlib.pyplot as plt


# read in files
X_train = pd.read_csv('train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])
X_train = X_train.values
y_train = y_train.values

# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Create a linear regression object: reg
svr = SVR()

# create the param grid

param_dist = {"kernel": ['rbf'],
              "gamma": expon(),
              "C": expon(),
              }

# create the randomized search grid

grid = RandomizedSearchCV(svr, param_dist, cv=5, scoring= ('r2', 'neg_root_mean_squared_error'), n_iter=50, refit=False)

grid.fit(X_train,y_train.ravel())

print(grid.best_params_, grid.best_score_)



scores, Cs, gammas = zip(*[(score.mean_test_neg_root_mean_squared_error, score.parameters['C'], score.parameters['gamma']) for score in grid.cv_results_])


'''
# create dataframe
df = pd.DataFrame(grid.cv_results_)

df.to_csv('svreg_full_features.csv')
'''

# make the plot

plt.scatter(np.log(Cs), np.log(gammas), s=50, c=scores, linewidths=0)
plt.xlabel("C")
plt.ylabel("gamma")
plt.show()