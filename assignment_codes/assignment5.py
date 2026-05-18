import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv('/home/user/Downloads/Social_Network_Ads.csv')

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

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

print("\nConfusion Matrix:\n", cm)
print(f"\nTP={TP}  TN={TN}  FP={FP}  FN={FN}")
print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision : {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall    : {recall_score(y_test, y_pred)*100:.2f}%")
print(f"Error Rate: {(1-accuracy_score(y_test, y_pred))*100:.2f}%")

