import pandas as pd
from sklearn.preprocessing import StandardScaler
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
    'sklearn.ensemble.GradientBoostingRegressor': {
        'learning_rate': [0.001,0.01,0.1,1,10],
        'n_estimators':[50,100,150,200,250,300],
    }}

tpot_optimizer = TPOTRegressor(generations=5, population_size=20, cv=5, verbosity=3, config_dict= tpot_config, n_jobs=1, max_time_mins=30, max_eval_time_mins=1)

# fit grid with data
tpot_optimizer.fit(X_train,y_train.ravel())

print("________BEST PIPELINE____________")
print(tpot_optimizer.fitted_pipeline_)

print("______PARETO FRONT___________")
print(tpot_optimizer.pareto_front_fitted_pipelines_)

print("________ALL MODELS BELOW____________")
print(tpot_optimizer.evaluated_individuals_)
