import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

X_train = pd.read_csv('train.csv')
X_train.columns = ["accommodates","bathrooms","bedrooms","calc_host_listings","guests_included","host_listings","latitude","longitude","minimum_nights","nr_of_reviews","d_centre","inst_bookable_f","inst_bookable_t","rt_Entire home","rt_Private","srt_Shared"]
print(X_train.head())

# Generate a mask for the upper triangle

cmap = sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)

sns.heatmap(X_train.corr(), center=0, cmap=cmap, linewidths=1, annot=False, fmt=".2f")

plt.show()

print(X_train.corr())

