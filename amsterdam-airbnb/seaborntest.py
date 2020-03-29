import pandas as pd
from seaborn import pairplot
import matplotlib.pyplot as plt

X_train = pd.read_csv('train.csv')

pairplot(X_train, hue='d_centre', kind='hist')
plt.show()