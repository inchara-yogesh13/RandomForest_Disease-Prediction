import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

# Load features
features = joblib.load("features.pkl")

# Load dataset for medicine lookup
df = pd.read_csv("training.csv")

patient = {feature: 0 for feature in features}

print("Enter symptoms one by one.")
print("Type 'done' when finished.\n")

while True:
    symptom = input("Symptom: ").strip().lower()

    if symptom == "done":
        break

    if symptom in patient:
        patient[symptom] = 1
    else:
        print("Symptom not found!")

# Convert to DataFrame
input_df = pd.DataFrame([patient])

# Predict disease
disease = model.predict(input_df)[0]

print("\nPredicted Disease:", disease)

# Confidence score
probability = model.predict_proba(input_df).max() * 100

print(f"Confidence: {probability:.2f}%")

# Find medicine
medicine = df[df["prognosis"] == disease]["medicine"].iloc[0]

print("Recommended Medicine:", medicine)