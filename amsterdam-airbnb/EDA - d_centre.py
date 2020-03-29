import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# read in csv files (only test data)
X_train = pd.read_csv('train.csv')
y_train = pd.read_csv('y_train.csv', names=['price'])


plt.scatter(x=X_train['d_centre'], y=y_train, alpha=0.4)
plt.show()

corcoeff = np.corrcoef(x=X_train['d_centre'], y=y_train, rowvar=False)
print(corcoeff)

