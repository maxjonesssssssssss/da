import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/home/user/Downloads/titanic.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nInfo:")
print(df.info())
print("\nStatistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

df['age']      = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(columns=['deck'])
print("\nAfter Cleaning:")
print(df.isnull().sum())

sns.countplot(data=df, x='survived')
plt.title('Survival Count')
plt.show()

sns.countplot(data=df, x='sex', hue='survived')
plt.title('Survival by Gender')
plt.show()

sns.countplot(data=df, x='pclass', hue='survived')
plt.title('Survival by Class')
plt.show()

sns.histplot(data=df, x='fare', bins=40, kde=True, color='coral')
plt.title('Fare Distribution')
plt.show()

print("""
=== OBSERVATIONS ===

1. SURVIVAL COUNT:
   - More passengers did NOT survive than survived.
   - Out of 891 passengers, only 342 survived.

2. SURVIVAL BY GENDER:
   - Females had much higher survival rate than males.
   - Most females survived while most males did not.
   - This is due to the 'women and children first' policy.

3. SURVIVAL BY CLASS:
   - 1st class passengers had the highest survival rate.
   - 3rd class passengers had the lowest survival rate.
   - Higher class = better access to lifeboats.

4. FARE DISTRIBUTION:
   - Fare is highly right skewed.
   - Most passengers paid less than 50 pounds.
   - Very few passengers paid extremely high fares (500+).
   - These high fare passengers were 1st class.
""")
