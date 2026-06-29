import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("training.csv")

# Features
X = df.drop(columns=["prognosis", "medicine"])

# Target
y = df["prognosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("="*40)
print("Random Forest Results")
print("="*40)

print("Accuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

# Save model
joblib.dump(model, "random_forest_model.pkl")

# Save feature names
joblib.dump(list(X.columns), "features.pkl")

print("\nModel Saved Successfully!")