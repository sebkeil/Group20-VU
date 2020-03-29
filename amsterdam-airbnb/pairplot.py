import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('test.csv')
df_select = df.loc[:,['accomodates','guests_included','number_of_reviews','d_centre']]

sns.pairplot(df_select)
plt.show()