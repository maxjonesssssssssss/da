import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/home/user/Downloads/titanic.csv')

print("Shape: ",df.shape)
print("\nFirst 5 row:")
print(df.head())
print("\nLast 5 row:")
print(df.tail())
print("\nInfo:")
print(df.info())
print("\nStatistics")
print(df.describe())
print("\nMissing values")
print(df.isnull().sum())

df['age']      = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(columns=['deck'])
print("\nAfter Cleaning:")
print(df.isnull().sum())

df['fare_norm'] = (df['fare'] - df['fare'].min()) / (df['fare'].max() - df['fare'].min())
df['age_norm']  = (df['age']  - df['age'].min())  / (df['age'].max()  - df['age'].min())

df['sex_encoded'] = df['sex'].map({'male': 0, 'female': 1})
dummies = pd.get_dummies(df['embarked'], prefix='embarked')
df = pd.concat([df, dummies], axis=1)

print("\nFinal columns:", df.columns.tolist())
print(df.head())
