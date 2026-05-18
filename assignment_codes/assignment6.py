import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
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
print("\nStatistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

X = df.drop(columns=['species'])
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

for i, cls in enumerate(model.classes_):
    TP = cm[i][i]
    FP = cm[:,i].sum() - TP
    FN = cm[i,:].sum() - TP
    TN = cm.sum() - TP - FP - FN
    print(f"{cls} -> TP={TP} TN={TN} FP={FP} FN={FN}")

print(f"\nAccuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision : {precision_score(y_test, y_pred, average='macro')*100:.2f}%")
print(f"Recall    : {recall_score(y_test, y_pred, average='macro')*100:.2f}%")
print(f"Error Rate: {(1-accuracy_score(y_test, y_pred))*100:.2f}%")

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title('Confusion Matrix - Naive Bayes')
plt.show()
