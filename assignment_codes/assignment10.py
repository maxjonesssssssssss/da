import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/home/user/Downloads/iris.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nInfo:")
print(df.info())
print("\nData Types:")
print(df.dtypes)
print("\nStatistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

cols = ['sepal_length','sepal_width','petal_length','petal_width']

print("""
=== FEATURE TYPES ===
sepal_length  -> Numeric (float)
sepal_width   -> Numeric (float)
petal_length  -> Numeric (float)
petal_width   -> Numeric (float)
species       -> Nominal (categorical)
""")

fig, axes = plt.subplots(2, 2, figsize=(10,7))
for i, col in enumerate(cols):
    axes.flatten()[i].hist(df[col], bins=20, edgecolor='black')
    axes.flatten()[i].set_title(col)
plt.suptitle('Histograms of All Features')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10,7))
for i, col in enumerate(cols):
    sns.boxplot(data=df, x='species', y=col, ax=axes.flatten()[i])
plt.suptitle('Boxplots by Species')
plt.tight_layout()
plt.show()

print("\nOutliers (IQR):")
for col in cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    out = df[(df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)]
    print(f"  {col}: {len(out)} outliers")
    
print("""
=== OBSERVATIONS ===

1. FEATURE TYPES:
   - 4 numeric features: sepal_length, sepal_width,
     petal_length, petal_width
   - 1 nominal feature: species (setosa, versicolor, virginica)

2. HISTOGRAMS:
   - sepal_length: roughly normal distribution
   - sepal_width: normal distribution, centered around 3.0
   - petal_length: bimodal (two peaks) — setosa is clearly
     separate from other two species
   - petal_width: bimodal — setosa has very small petal width

3. BOXPLOTS:
   - setosa has smallest petal length and width
   - virginica has largest petal length and width
   - versicolor falls in between setosa and virginica
   - sepal_width has visible outliers in setosa group

4. OUTLIERS:
   - sepal_width has the most outliers
   - petal_length and petal_width have very few outliers
   - Outliers appear as dots outside whiskers in boxplots

5. DISTRIBUTION COMPARISON:
   - setosa is clearly separable from other two species
   - versicolor and virginica overlap in sepal measurements
   - petal features are better for separating species
     than sepal features

6. BEST FEATURES FOR CLASSIFICATION:
   - petal_length and petal_width are most useful
   - sepal features have more overlap between species
""")
