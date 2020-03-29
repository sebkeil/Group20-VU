from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

X_train = pd.read_csv('train.csv')

m = TSNE(learning_rate=50)
tsne_features = m.fit_transform(X_train)
print(tsne_features.shape)

