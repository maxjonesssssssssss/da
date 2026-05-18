import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
df = pd.DataFrame({
    'study_hours'  : np.round(np.random.normal(5, 2, 100), 1),
    'math_score'   : np.random.randint(40, 100, 100),
    'attendance'   : np.round(np.random.uniform(50, 100, 100), 1),
    'science_score': np.random.randint(40, 100, 100),
})

print("Shape: ",df.shape)
print("\nFirst 5 row:")
print(df.head())
print("\nLast 5 row:")
print(df.tail())
print("\nInfo:")
print(df.info())
print("\nStatistics")
print(df.describe())

df.loc[0:5, 'study_hours'] = np.nan
df.loc[0, 'study_hours'] = 25   
df.loc[1, 'math_score']  = 200    


print("Missing values:\n", df.isnull().sum())

df.loc[df['study_hours'] > 16, 'study_hours'] = np.nan
df.loc[df['math_score']  > 100, 'math_score'] = np.nan


df = df.fillna(df.median())
print("\nAfter fixing:\n", df.isnull().sum())


for col in df.columns:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    out = df[(df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)]
    print(f"{col}: {len(out)} outliers")


for col in df.columns:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1-1.5*IQR, Q3+1.5*IQR)

df['study_log']  = np.log(df['study_hours'] + 0.1)
df['math_std']   = (df['math_score'] - df['math_score'].mean()) / df['math_score'].std()
df['att_norm']   = (df['attendance'] - df['attendance'].min()) / (df['attendance'].max() - df['attendance'].min())
df['sci_sqrt']   = np.sqrt(df['science_score'])

print("\nTransformations done!")
print(df.head())

df[['study_hours','math_score','attendance','science_score']].boxplot(figsize=(8,5))
plt.title('Boxplot')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].hist(df['study_hours'], bins=15, color='coral', edgecolor='black')
axes[0].set_title('BEFORE Log')
axes[1].hist(df['study_log'],   bins=15, color='green', edgecolor='black')
axes[1].set_title('AFTER Log')
plt.show()
