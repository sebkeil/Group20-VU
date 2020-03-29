from tpot import TPOTRegressor
import pandas as pd

X_train = pd.read_csv('train.csv')
X_train = X_train.values
y_train = pd.read_csv('y_train.csv', names=['price'])
y_train = y_train.values

X_test = pd.read_csv('test.csv')
X_test = X_test.values
y_test = pd.read_csv('y_test.csv', names=['price'])
y_test = y_test.values

tpot = TPOTRegressor(verbosity=3, max_time_mins=10)
tpot.fit(X_train,y_train.ravel())

print(tpot.score(X_test,y_test.ravel()))

# tpot.fitted_pipeline_

tpot.export('tpot_pipeline.py')
print(tpot.pareto_front_fitted_pipelines_)
print(tpot.fitted_pipeline_.steps[-1][1].feature_importances_)
