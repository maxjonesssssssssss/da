import pandas as pd
import numpy as np

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

print("\n--- Age grouped by Pclass ---")
print(df.groupby('pclass')['age'].agg(
    Mean='mean', Median='median', Std='std', Min='min', Max='max'
).round(2))

print("\n--- Fare grouped by Pclass ---")
print(df.groupby('pclass')['fare'].agg(
    Mean='mean', Median='median', Std='std', Min='min', Max='max'
).round(2))

df['pclass_num'] = df['pclass'].map({1:1, 2:2, 3:3})
print("\nPclass numeric list:", df['pclass_num'].tolist()[:20])

df2 = pd.read_csv('/home/user/Downloads/iris.csv')

print("\n\nIris Shape:", df2.shape)
print("\nFirst 5 rows:")
print(df2.head())
print("\nLast 5 row:")
print(df2.tail())
print("\nInfo:")
print(df2.info())
print("\nStatistics:")
print(df2.describe())
print("\nMissing values")
print(df.isnull().sum())

for species in df2['species'].unique():
    print(f"\n--- {species} ---")
    subset = df2[df2['species'] == species]
    for col in ['sepal_length','sepal_width','petal_length','petal_width']:
        print(f"  {col}: mean={round(subset[col].mean(),2)}, median={round(subset[col].median(),2)}, std={round(subset[col].std(),2)}, 25th={round(subset[col].quantile(0.25),2)}, 75th={round(subset[col].quantile(0.75),2)}")
