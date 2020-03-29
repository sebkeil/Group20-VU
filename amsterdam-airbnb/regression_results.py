import sklearn.metrics as metrics
from math import sqrt

def regression_results(y_test, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_test, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred)
    mse=metrics.mean_squared_error(y_test, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_test, y_pred)
    r2=metrics.r2_score(y_test, y_pred)

    print('explained_variance: ', round(explained_variance,2))
    print('mean_squared_log_error: ', round(mean_squared_log_error,2))
    print('r2: ', round(r2,2))
    print('mean_absolute_error: ', round(mean_absolute_error,2))
    print('MSE: ', round(mse,2))
    print('RMSE: ', round(sqrt(mse),2))
    print('median_absolute_error:', round(median_absolute_error,2))
