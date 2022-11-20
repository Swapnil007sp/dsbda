import pandas as pd
import numpy as np

#df = pd.read_csv("iris.data",header=None)

col_names= ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
df = pd.read_csv("iris.data",names=col_names)
print(df.head())

print(df.head(n=5))

print(df.tail(n=5))

print(df.index)

#print(df.columns)

print(df.shape)
#
#print(df.dtypes)

#print(df.columns.values)



print(df[col_names])

print(df.sort_index(axis=1,ascending=False))

print(df.sort_values(by=col_names))

print(df.iloc[5])

print(df[0:3])

print(df.isnull())

print(df.isnull().any())

print(df.isnull().sum().sum())

print(df.isnull().sum(axis=1))

print(df.isnull().sum())

from sklearn import preprocessing

print(df['Species'].unique())

one_hot_df=pd.get_dummies(df,prefix="Species",columns=['Species'],drop_first=False)
print(one_hot_df)
print(df.describe(include='all'))