import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
df = pd.read_csv("data/gesture_data.csv", header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, stratify=y, random_state=42
)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))
print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions))
joblib.dump(clf, "models/gesture_rf.pkl")
print("Saved model to models/gesture_rf.pkl")
