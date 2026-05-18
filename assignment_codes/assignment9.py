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

plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='sex', y='age', hue='survived',
            palette={0:'red', 1:'green'})
plt.title('Age by Gender and Survival')
plt.legend(title='survived', labels=['No','Yes'])
plt.show()

print("\nAge stats by Gender and Survival:")
print(df.groupby(['sex','survived'])['age'].describe().round(2))

print("""
=== OBSERVATIONS ===

1. FEMALE SURVIVORS (green):
   - Median age of survived females is around 28 years.
   - Most survived females were between 20 to 40 years old.
   - Females had very high survival rate overall.

2. FEMALE NON SURVIVORS (red):
   - Median age of non survived females is around 25 years.
   - Slightly younger than survived females.
   - Even young females had lower survival chances than older ones.

3. MALE SURVIVORS (green):
   - Survived males tend to be younger.
   - Median age around 27 years.
   - Younger males had better survival chances.

4. MALE NON SURVIVORS (red):
   - Median age around 28 years.
   - Many outliers at ages 60 and above (dots above whiskers).
   - Older males had very low survival chances.

5. OUTLIERS:
   - Dots above whiskers represent very old passengers (70-80 years).
   - Most outliers appear in male non survivor group.

6. OVERALL:
   - Females survived much more than males.
   - This confirms the 'women and children first' evacuation policy.
   - Age played a bigger role in male survival than female survival.
""")
