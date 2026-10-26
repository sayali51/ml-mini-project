"""
View the contents of the saved model/preprocessing files.
Run this in the same folder as your .pkl files:
    python view_saved_files.py
Screenshot the printed output for your report.
"""

import joblib

model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")
le_sex = joblib.load("le_sex.pkl")
le_embarked = joblib.load("le_embarked.pkl")
le_title = joblib.load("le_title.pkl")
model_accuracy = joblib.load("model_accuracy.pkl")

print("=" * 55)
print("SAVED MODEL DETAILS")
print("=" * 55)
print(f"Model type       : {type(model).__name__}")
print(f"Model parameters : {model.get_params()}")
print(f"Features used    : {list(model.feature_names_in_)}")
print(f"Test Accuracy    : {model_accuracy * 100:.2f}%")

print("\n" + "=" * 55)
print("SAVED ENCODERS")
print("=" * 55)
print(f"Sex classes      : {list(le_sex.classes_)}  ->  {list(range(len(le_sex.classes_)))}")
print(f"Embarked classes : {list(le_embarked.classes_)}  ->  {list(range(len(le_embarked.classes_)))}")
print(f"Title classes    : {list(le_title.classes_)}  ->  {list(range(len(le_title.classes_)))}")

print("\n" + "=" * 55)
print("SAVED SCALER (Age, Fare, FamilySize)")
print("=" * 55)
print(f"Mean : {scaler.mean_}")
print(f"Scale (std dev) : {scaler.scale_}")