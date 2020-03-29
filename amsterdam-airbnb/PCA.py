import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

# drop features after feature selection
X_train = X_train.drop(['bathrooms', 'bedrooms', 'calculated_host_listings_count', 'guests_included', 'instant_bookable_f','instant_bookable_t','room_type_Entire home/apt','room_type_Private room','room_type_Shared room'], axis=1)
X_test = X_test.drop(['bathrooms', 'bedrooms','calculated_host_listings_count','guests_included', 'instant_bookable_f','instant_bookable_t','room_type_Entire home/apt','room_type_Private room','room_type_Shared room'], axis=1)

# center(mu=0) and scale (std=1) the data
scaled_X_train= preprocessing.scale(X_train)
scaled_X_test = preprocessing.scale(X_test)

print(scaled_X_train)

# generate pca object
pca = PCA()

#generate loading scores and scaled data
pca.fit(scaled_X_train)
pca.fit(scaled_X_test)

# generate coordinates for pca graph
pca_X_train_data = pca.transform(scaled_X_train)
pca_X_test_data = pca.transform(scaled_X_test)

# calculate the percentage of variation that each pca accounts for
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

# create the labels for the scree plot
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

# create the scree plot
''''
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
'''
# put the pca data into a dataframe
pca_df_X_train = pd.DataFrame(pca_X_train_data, columns=labels)
pca_df_X_test = pd.DataFrame(pca_X_test_data, columns=labels)

'''
# create scatter plots
plt.scatter(pca_df.PC1, pca_df.PC2, alpha=0.3)
plt.title('PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.show()
'''
# get loading scores
loading_scores = pd.Series(pca.components_[0])
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_10_houses = sorted_loading_scores[0:10].index.values

print(loading_scores)

#print(loading_scores[top_10_houses])

pca_df_X_train.to_csv('pca_df_X_train.csv')
pca_df_X_test.to_csv('pca_df_X_test.csv')

# make a paiplot
'''
sns.pairplot(pca_df_X_train)
plt.show()
'''

print(pca.explained_variance_ratio_.cumsum())

# see how much each component affects the construction of the principal component
print(pca.components_)