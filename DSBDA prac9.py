import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/swapnil/Desktop/titanic-train.csv')
df.head()

print(df.columns)

print(df.info())

print(df.describe())

print(df.isnull().sum())

sns.boxplot(df['Sex'],df['Age'])

sns.boxplot(df['Sex'],df['Age'], df['Survived'])

plt.show()
